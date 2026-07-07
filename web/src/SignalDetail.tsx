import { useEffect, useRef, useState } from 'react';
import {
  ColorType, LineStyle, createChart, type IChartApi, type UTCTimestamp,
} from 'lightweight-charts';
import { get } from './api';
import { fmtMoney, fmtPct, fmtPrice, fmtR } from './format';
import { Chip, ConfBar, Dot, RegimeChip, SideTag } from './ui';
import type { CandlesResponse, Plan } from './types';

const BLOCK_TITLES = ['Thesis', 'Why it may work', 'Why it may lose', 'Caveats'];

const CHART_LONG = '#10b981';
const CHART_SHORT = '#f43f5e';
const CHART_PRIMARY = '#22d3ee';
const CHART_SR = 'rgba(148,163,184,0.35)';
const CHART_LIQ = '#be123c';
const CHART_EMA20 = 'rgba(56,189,248,0.65)';
const CHART_EMA50 = 'rgba(129,140,248,0.65)';
const CHART_EMA200 = 'rgba(148,163,184,0.55)';
const CHART_VWAP = 'rgba(251,191,36,0.8)';
const CHART_BB = 'rgba(148,163,184,0.24)';
const CHART_RSI = 'rgba(168,139,250,0.9)';

export default function SignalDetail({ symbol, tf, plan, onClose }:
  { symbol: string; tf: string; plan?: Plan; onClose: () => void }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const [data, setData] = useState<CandlesResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    setData(null);
    setErr(null);
    get<CandlesResponse>(`/api/candles?symbol=${encodeURIComponent(symbol)}&tf=${tf}&limit=300`)
      .then(setData)
      .catch(e => setErr(String(e)));
  }, [symbol, tf]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !data?.candles?.length) return;

    const chart = createChart(el, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#64748b',
        fontFamily: '"JetBrains Mono", monospace',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(148,163,184,0.06)' },
        horzLines: { color: 'rgba(148,163,184,0.06)' },
      },
      rightPriceScale: { borderColor: 'rgba(148,163,184,0.15)' },
      timeScale: { borderColor: 'rgba(148,163,184,0.15)', timeVisible: true },
      crosshair: { mode: 0 },
      autoSize: true,
    });
    chartRef.current = chart;

    // Adaptive price precision — a 2-decimal axis renders sub-cent alts as 0.00.
    const lastClose = data.candles[data.candles.length - 1].close;
    const precision = lastClose >= 100 ? 2 : lastClose >= 1 ? 4
      : lastClose >= 0.01 ? 6 : 8;
    const candles = chart.addCandlestickSeries({
      upColor: CHART_LONG, downColor: CHART_SHORT, borderVisible: false,
      wickUpColor: CHART_LONG, wickDownColor: CHART_SHORT,
      priceFormat: { type: 'price', precision, minMove: 10 ** -precision },
    });
    candles.setData(data.candles.map(c => ({
      time: c.time as UTCTimestamp, open: c.open, high: c.high,
      low: c.low, close: c.close,
    })));

    const vol = chart.addHistogramSeries({
      priceFormat: { type: 'volume' }, priceScaleId: 'vol',
      color: 'rgba(100,116,139,0.35)',
    });
    chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    vol.setData(data.candles.map(c => ({
      time: c.time as UTCTimestamp, value: c.volume,
      color: c.close >= c.open ? 'rgba(16,185,129,0.35)' : 'rgba(244,63,94,0.35)',
    })));

    // Indicator overlays (EMA ribbon, VWAP, Bollinger band) — the very series
    // that produced the signal, drawn faint so the candles stay legible.
    const addLine = (points: { time: number; value: number }[] | undefined,
                     color: string, width = 1, style: LineStyle = LineStyle.Solid) => {
      if (!points?.length) return;
      const s = chart.addLineSeries({
        color, lineWidth: width as 1 | 2 | 3, lineStyle: style,
        priceLineVisible: false, lastValueVisible: false,
        crosshairMarkerVisible: false,
      });
      s.setData(points.map(p => ({ time: p.time as UTCTimestamp, value: p.value })));
    };
    const ov = data.overlays;
    if (ov) {
      addLine(ov.bb_up, CHART_BB);
      addLine(ov.bb_low, CHART_BB);
      addLine(ov.ema200, CHART_EMA200);
      addLine(ov.ema50, CHART_EMA50);
      addLine(ov.ema20, CHART_EMA20);
      addLine(ov.vwap, CHART_VWAP, 1, LineStyle.Dashed);
    }

    // RSI oscillator in its own sub-pane below price, with 30/70 guides.
    if (ov?.rsi?.length) {
      chart.priceScale('right').applyOptions({ scaleMargins: { top: 0.04, bottom: 0.42 } });
      const rsi = chart.addLineSeries({
        color: CHART_RSI, lineWidth: 1, priceScaleId: 'rsi',
        priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false,
      });
      chart.priceScale('rsi').applyOptions({ scaleMargins: { top: 0.62, bottom: 0.2 } });
      rsi.setData(ov.rsi.map(p => ({ time: p.time as UTCTimestamp, value: p.value })));
      rsi.createPriceLine({ price: 70, color: 'rgba(244,63,94,0.4)', lineWidth: 1,
        lineStyle: LineStyle.Dotted, title: 'RSI 70' });
      rsi.createPriceLine({ price: 30, color: 'rgba(16,185,129,0.4)', lineWidth: 1,
        lineStyle: LineStyle.Dotted, title: '30' });
    }

    // Detected chart patterns as labeled markers (one per bar, bias-colored).
    if (data.patterns?.length) {
      const seen = new Set<number>();
      const markers = [...data.patterns]
        .sort((a, b) => a.time - b.time)
        .filter(p => !seen.has(p.time) && seen.add(p.time))
        .map(p => ({
          time: p.time as UTCTimestamp,
          position: (p.bias === 'bearish' ? 'aboveBar' : 'belowBar') as 'aboveBar' | 'belowBar',
          color: p.bias === 'bearish' ? CHART_SHORT
            : p.bias === 'bullish' ? CHART_LONG : CHART_SR,
          shape: (p.bias === 'bearish' ? 'arrowDown' : 'arrowUp') as 'arrowDown' | 'arrowUp',
          text: p.name.replace(/_/g, ' '),
        }));
      candles.setMarkers(markers);
    }

    for (const lvl of data.sr_levels.slice(0, 8)) {
      candles.createPriceLine({
        price: lvl.price, color: CHART_SR, lineWidth: 1,
        lineStyle: LineStyle.Dotted, title: `S/R ×${lvl.touches}`,
      });
    }
    if (plan) {
      candles.createPriceLine({
        price: plan.entry, color: CHART_PRIMARY, lineWidth: 2,
        lineStyle: LineStyle.Solid, title: 'entry',
      });
      candles.createPriceLine({
        price: plan.stop_loss, color: CHART_SHORT, lineWidth: 2,
        lineStyle: LineStyle.Dashed, title: 'stop',
      });
      plan.take_profits.forEach((tp, i) => {
        candles.createPriceLine({
          price: tp.price, color: CHART_LONG, lineWidth: 2,
          lineStyle: LineStyle.Dashed, title: `TP${i + 1}`,
        });
      });
      candles.createPriceLine({
        price: plan.liquidation_price, color: CHART_LIQ, lineWidth: 1,
        lineStyle: LineStyle.SparseDotted, title: 'LIQ',
      });
    }
    chart.timeScale().fitContent();

    return () => { chart.remove(); chartRef.current = null; };
  }, [data, plan]);

  const quote = symbol.split('/')[1]?.split(':')[0];

  return (
    <div className="fixed inset-0 z-30 flex justify-end bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="flex h-full w-full max-w-5xl flex-col overflow-y-auto border-l border-line bg-surface-1 shadow-2xl dark:bg-slate-950"
        onClick={e => e.stopPropagation()}
      >
        <header className="sticky top-0 z-10 flex flex-wrap items-center gap-3 border-b border-line bg-surface-1/95 px-5 py-3 backdrop-blur dark:bg-slate-900/95">
          <div className="flex items-baseline gap-2">
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">{symbol.split('/')[0]}</h2>
            {quote && <span className="text-sm text-slate-500">/{quote}</span>}
          </div>
          <Chip>{tf}</Chip>
          {plan && <SideTag side={plan.side} />}
          {plan && <RegimeChip regime={plan.regime} />}
          {plan && <ConfBar value={plan.confidence} />}
          <button onClick={onClose}
            className="ml-auto inline-flex h-8 w-8 items-center justify-center rounded-lg text-slate-500 transition-colors hover:bg-surface-2 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100">
            ✕
          </button>
        </header>

        <div className="grid gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_340px]">
          <div className="space-y-3">
            <div className="card h-[420px] p-2" ref={containerRef}>
              {!data && !err && <div className="p-4 text-2xs text-slate-500">loading candles…</div>}
              {err && <div className="p-4 text-2xs text-danger">{err}</div>}
            </div>
            <ChartLegend />
          </div>

          <aside className="space-y-4">
            {plan ? (
              <>
                <div className="card p-4">
                  <div className="mb-3 text-2xs font-semibold uppercase tracking-wider text-slate-500">Risk metrics</div>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                    <KV k="entry" v={fmtPrice(plan.entry)} />
                    <KV k="stop" v={fmtPrice(plan.stop_loss)} tone="danger" />
                    {plan.take_profits.map((tp, i) => (
                      <KV key={i} k={`TP${i + 1} (${fmtPct(tp.alloc, 0)})`}
                        v={`${fmtPrice(tp.price)} · ${tp.r.toFixed(1)}R`} tone="success" />
                    ))}
                    <KV k="liquidation" v={fmtPrice(plan.liquidation_price)} tone="danger" />
                    <KV k="leverage" v={`${plan.leverage.toFixed(1)}x`} />
                    <KV k="qty" v={plan.qty.toPrecision(4)} />
                    <KV k="notional" v={fmtMoney(plan.notional)} />
                    <KV k="margin" v={fmtMoney(plan.margin)} />
                    <KV k="risk" v={fmtMoney(plan.risk_amount)} />
                    <KV k="fees est." v={fmtMoney(plan.fees_estimate)} />
                    <KV k="EV" v={fmtR(plan.expected_value_r)}
                      tone={plan.expected_value_r >= 0 ? 'success' : 'danger'} />
                    <KV k="win rate" v={fmtPct(plan.expected_win_rate, 0)} />
                  </div>
                </div>

                {plan.ml_prob !== null && (
                  <div className="card p-4">
                    <div className="text-2xs font-semibold uppercase tracking-wider text-slate-500">Meta-model</div>
                    <div className="num mt-1 text-slate-900 dark:text-slate-200">{fmtPct(plan.ml_prob, 0)}
                      <span className="ml-2 text-2xs text-slate-500">
                        P(win) · blended at {fmtPct(plan.ml_weight, 0)}
                      </span>
                    </div>
                    {plan.ml_ev_r != null && (
                      <div className="num mt-1 text-sm">
                        <span className={plan.ml_ev_r >= 0 ? 'text-success' : 'text-danger'}>
                          {fmtR(plan.ml_ev_r)}
                        </span>
                        <span className="ml-2 text-2xs text-slate-500">predicted E[R] · reward head</span>
                      </div>
                    )}
                    {plan.ml_contribs.length > 0 && (
                      <div className="mt-2 text-2xs text-slate-500">
                        top drivers: <span className="text-slate-400">{plan.ml_contribs.join(', ')}</span>
                      </div>
                    )}
                  </div>
                )}

                {plan.warnings.length > 0 && (
                  <div className="card border-warning/30 p-4">
                    <div className="flex items-center gap-2 text-2xs font-semibold uppercase tracking-wider text-warning">
                      <Dot tone="warning" /> warnings
                    </div>
                    <ul className="mt-2 list-inside list-disc space-y-1 text-2xs text-slate-600 dark:text-slate-300">
                      {plan.warnings.map((w, i) => <li key={i}>{w}</li>)}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <div className="card p-4 text-2xs text-slate-500">
                No active signal for this symbol — chart only. S/R levels are drawn
                from touch-counted pivots.
              </div>
            )}
          </aside>
        </div>

        {plan && (
          <div className="space-y-4 px-5 pb-8">
            <div className="grid gap-4 md:grid-cols-2">
              {plan.explanation.map((block, i) => (
                <div key={i} className="card p-4">
                  <div className="text-2xs font-semibold uppercase tracking-wider text-primary">
                    {BLOCK_TITLES[i] ?? `Note ${i + 1}`}
                  </div>
                  <p className="mt-2 leading-relaxed text-slate-600 dark:text-slate-300">{block}</p>
                </div>
              ))}
            </div>
            {plan.rationale.length > 0 && (
              <details className="card overflow-hidden">
                <summary className="cursor-pointer bg-surface-2/50 px-4 py-3 text-2xs font-semibold uppercase tracking-wider text-slate-500 dark:bg-slate-800/40">
                  raw evidence ({plan.rationale.length})
                </summary>
                <ul className="num space-y-1 px-4 pb-4 pt-3 text-2xs text-slate-500">
                  {plan.rationale.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ChartLegend() {
  const items = [
    { color: CHART_EMA20, label: 'EMA20' },
    { color: CHART_EMA50, label: 'EMA50' },
    { color: CHART_EMA200, label: 'EMA200' },
    { color: CHART_VWAP, label: 'VWAP', dashed: true },
    { color: CHART_BB, label: 'Bollinger' },
    { color: CHART_RSI, label: 'RSI' },
    { color: CHART_PRIMARY, label: 'entry' },
    { color: CHART_SHORT, label: 'stop', dashed: true },
    { color: CHART_LONG, label: 'take profit', dashed: true },
    { color: CHART_SR, label: 'S/R' },
    { color: CHART_LIQ, label: 'liquidation' },
  ];
  return (
    <div className="flex flex-wrap gap-3 px-1 text-2xs text-slate-500">
      {items.map(item => (
        <span key={item.label} className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-0.5 w-4"
            style={{ background: item.color, borderTop: item.dashed ? '2px dashed' : undefined }}
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

function KV({ k, v, tone }: { k: string; v: string; tone?: 'success' | 'danger' | 'warning' }) {
  const color = tone === 'success' ? 'text-success'
    : tone === 'danger' ? 'text-danger'
    : tone === 'warning' ? 'text-warning'
    : 'text-slate-900 dark:text-slate-200';
  return (
    <div className="contents">
      <span className="text-2xs uppercase tracking-wide text-slate-500">{k}</span>
      <span className={`num text-right ${color}`}>{v}</span>
    </div>
  );
}
