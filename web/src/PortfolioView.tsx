import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ColorType, createChart, type UTCTimestamp,
} from 'lightweight-charts';
import ExposureHeatmap from './ExposureHeatmap';
import { useTick } from './api';
import { fmtMoney, fmtPct, fmtPrice, fmtSigned, timeAgo } from './format';
import { Chip, Dot, Empty, Section, SideTag, Skeleton, Stat } from './ui';
import type { ClosedTrade, Exposure, OpenPosition, Positions } from './types';

type SortKey = 'closed_at' | 'pnl' | 'r' | 'symbol';
type SortDir = 'asc' | 'desc';
type Toast = { message: string; tone: 'success' | 'danger' } | null;

export default function PortfolioView({ positions, exposure, onClose }:
  { positions: Positions | null; exposure: Exposure | null; onClose: (symbol: string) => Promise<void> }) {
  const now = useTick(5000);
  const [toast, setToast] = useState<Toast>(null);

  useEffect(() => {
    if (!toast) return;
    const id = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(id);
  }, [toast?.message]);

  if (!positions) {
    return (
      <div className="space-y-5">
        <Skeleton rows={2} />
        <Section title="Equity curve"><Skeleton rows={4} /></Section>
      </div>
    );
  }

  const dayPnl = positions.mtm_equity - (positions.day_anchor?.equity ?? positions.mtm_equity);
  const unreal = positions.open_positions.reduce((s, p) => s + p.unrealized_pnl, 0);

  return (
    <div className="space-y-5">
      {toast && (
        <div className={`rounded-lg border px-4 py-2 text-sm ${
          toast.tone === 'success'
            ? 'border-success/30 bg-success-dim text-success'
            : 'border-danger/30 bg-danger-dim text-danger'
        }`}>
          {toast.message}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="equity (MTM)" value={fmtMoney(positions.mtm_equity)}
          sub={`realized ${fmtMoney(positions.equity)}`} />
        <Stat label="return" value={fmtPct(positions.return_pct)}
          tone={positions.return_pct >= 0 ? 'success' : 'danger'} />
        <Stat label="today" value={fmtSigned(dayPnl)}
          tone={dayPnl >= 0 ? 'success' : 'danger'} sub={positions.day_anchor?.date} />
        <Stat label="open / unrealized" value={positions.open_positions.length}
          sub={`${fmtSigned(unreal)} USDT`} />
        <Stat label="losing streak" value={positions.consecutive_losses}
          tone={positions.consecutive_losses >= 3 ? 'warning' : undefined}
          sub={positions.consecutive_losses >= 3 ? 'approaching breaker' : 'breaker at 5'} />
      </div>

      <ExposureHeatmap exposure={exposure} />

      <Section title="Equity curve (mark-to-market)">
        {positions.equity_curve.length < 2
          ? <Empty title="Curve appears after the first tracked candles" />
          : <EquityCurveCard curve={positions.equity_curve} />}
      </Section>

      <Section title={`Open positions — ${positions.open_positions.length}`}>
        {positions.open_positions.length === 0 ? (
          <Empty title="Flat." hint="The scanner opens positions when ranked setups pass every cap and breaker." />
        ) : (
          <div className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-3">
            {positions.open_positions.map(p => (
              <PositionCard key={p.id} p={p} now={now} onClose={onClose} onToast={setToast} />
            ))}
          </div>
        )}
      </Section>

      <Section title={`Closed trades — ${positions.closed_trades.length} total`}>
        {positions.closed_trades.length === 0 ? (
          <Empty title="No closed trades yet" />
        ) : (
          <ClosedTradesTable trades={positions.closed_trades} now={now} />
        )}
      </Section>
    </div>
  );
}

function EquityCurveCard({ curve }: { curve: [string, number][] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const chart = createChart(el, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#64748b', fontFamily: '"JetBrains Mono", monospace', fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(148,163,184,0.06)' },
        horzLines: { color: 'rgba(148,163,184,0.06)' },
      },
      rightPriceScale: { borderColor: 'rgba(148,163,184,0.15)' },
      timeScale: { borderColor: 'rgba(148,163,184,0.15)', timeVisible: true },
      autoSize: true,
    });
    const series = chart.addAreaSeries({
      lineColor: 'var(--primary)', lineWidth: 2,
      topColor: 'rgba(34,211,238,0.25)', bottomColor: 'rgba(34,211,238,0.0)',
    });
    // dedupe timestamps (chart requires ascending unique times)
    const seen = new Set<number>();
    const data = [];
    for (const [iso, v] of curve) {
      const t = Math.floor(new Date(iso).getTime() / 1000);
      if (!isNaN(t) && !seen.has(t)) {
        seen.add(t);
        data.push({ time: t as UTCTimestamp, value: v });
      }
    }
    data.sort((a, b) => (a.time as number) - (b.time as number));
    series.setData(data);
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [curve]);

  const start = curve[0]?.[1] ?? 0;
  const end = curve[curve.length - 1]?.[1] ?? 0;
  const change = end - start;

  return (
    <div className="card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500">Equity</div>
          <div className={`num text-xl font-semibold ${change >= 0 ? 'text-success' : 'text-danger'}`}>
            {fmtMoney(end)}
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xs font-semibold uppercase tracking-wider text-neutral-500">Total change</div>
          <div className={`num font-semibold ${change >= 0 ? 'text-success' : 'text-danger'}`}>
            {fmtSigned(change)} ({fmtPct(change / Math.max(start, 1))})
          </div>
        </div>
      </div>
      <div ref={ref} className="h-60 w-full rounded-lg bg-surface-2/50 dark:bg-neutral-800/30" />
    </div>
  );
}

function PositionCard({ p, now, onClose, onToast }:
  { p: OpenPosition; now: number; onClose: (symbol: string) => Promise<void>;
    onToast: (t: Toast) => void }) {
  const tpTotal = p.take_profits.length;
  const tpFilled = p.take_profits.filter(tp => tp[3]).length;
  const [busy, setBusy] = useState(false);

  const handleClose = async () => {
    if (!confirm(`Close open position ${p.symbol}?`)) return;
    setBusy(true);
    try {
      await onClose(p.symbol);
      onToast({ message: `Closed ${p.symbol}`, tone: 'success' });
    } catch (e) {
      onToast({ message: `Close failed: ${(e as Error).message}`, tone: 'danger' });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card flex flex-col gap-3 p-4">
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-base font-semibold text-neutral-900 dark:text-neutral-100">{p.symbol.split('/')[0]}</span>
            <span className="text-2xs text-neutral-500">{p.timeframe}</span>
          </div>
          <div className="mt-1 flex items-center gap-2">
            <SideTag side={p.side} />
            <Chip>{p.leverage.toFixed(1)}x</Chip>
            {p.breakeven_moved && <Chip tone="accent" title="stop moved to breakeven">BE</Chip>}
          </div>
        </div>
        <div className="text-right">
          <div className={`num font-semibold ${p.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}`}>
            {fmtSigned(p.unrealized_pnl)}
          </div>
          <div className="text-2xs text-neutral-500">uPnL USDT</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-sm">
        <KV k="entry" v={fmtPrice(p.entry)} />
        <KV k="last" v={fmtPrice(p.last_price)} />
        <KV k="stop" v={fmtPrice(p.stop)} />
        <KV k="margin" v={fmtMoney(p.margin)} />
        <KV k="rPnL" v={fmtSigned(p.realized_pnl)} tone={p.realized_pnl >= 0 ? 'success' : 'danger'} />
        <KV k="opened" v={timeAgo(p.opened_at, now)} />
      </div>

      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-2xs text-neutral-500" title="take-profit ladder fills">
          <span>TP ladder</span>
          {p.take_profits.map((tp, i) => (
            <span key={i} className={tp[3] ? 'text-success' : 'text-neutral-600'}>●</span>
          ))}
          <span className="ml-1">{tpFilled}/{tpTotal}</span>
        </div>
        <button
          onClick={handleClose}
          disabled={busy}
          className="rounded-md border border-danger/30 bg-danger-dim px-2.5 py-1 text-2xs font-medium text-danger transition-colors hover:bg-danger/20 disabled:opacity-50"
        >
          {busy ? '…' : 'Close'}
        </button>
      </div>
    </div>
  );
}

function KV({ k, v, tone }:
  { k: string; v: string; tone?: 'success' | 'danger' }) {
  const color = tone === 'success' ? 'text-success'
    : tone === 'danger' ? 'text-danger'
    : 'text-neutral-900 dark:text-neutral-200';
  return (
    <div className="flex justify-between rounded-md bg-surface-2 px-2 py-1 dark:bg-neutral-800/50">
      <span className="text-2xs text-neutral-500">{k}</span>
      <span className={`num text-xs font-medium ${color}`}>{v}</span>
    </div>
  );
}

function ClosedTradesTable({ trades, now }: { trades: ClosedTrade[]; now: number }) {
  const [sort, setSort] = useState<SortKey>('closed_at');
  const [dir, setDir] = useState<SortDir>('desc');
  const [filter, setFilter] = useState<'all' | 'win' | 'loss'>('all');

  const filtered = useMemo(() => {
    let list = [...trades];
    if (filter === 'win') list = list.filter(t => t.win);
    if (filter === 'loss') list = list.filter(t => !t.win);
    list.sort((a, b) => {
      let va: number | string = a[sort];
      let vb: number | string = b[sort];
      if (sort === 'closed_at') {
        va = new Date(a.closed_at).getTime();
        vb = new Date(b.closed_at).getTime();
      }
      if (va < vb) return dir === 'asc' ? -1 : 1;
      if (va > vb) return dir === 'asc' ? 1 : -1;
      return 0;
    });
    return list.slice(0, 50);
  }, [trades, sort, dir, filter]);

  const header = (key: SortKey, label: string) => (
    <th
      className="th cursor-pointer select-none hover:text-neutral-400"
      onClick={() => {
        if (sort === key) setDir(d => d === 'asc' ? 'desc' : 'asc');
        else { setSort(key); setDir('desc'); }
      }}
    >
      {label} {sort === key && (dir === 'asc' ? '↑' : '↓')}
    </th>
  );

  return (
    <div className="p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {(['all', 'win', 'loss'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-md border px-2.5 py-1 text-2xs font-medium capitalize transition-colors ${
              filter === f
                ? 'border-primary/30 bg-primary-dim text-primary'
                : 'border-line bg-surface-2 text-neutral-500 hover:text-neutral-700 dark:bg-neutral-800 dark:hover:text-neutral-300'
            }`}
          >
            {f}
          </button>
        ))}
        <span className="ml-auto text-2xs text-neutral-500">
          showing {filtered.length} of {trades.length}
        </span>
      </div>

      <div className="table-responsive max-h-96 rounded-lg border border-line">
        <table className="w-full whitespace-nowrap text-sm">
          <thead className="sticky top-0 z-10 bg-surface-1 dark:bg-neutral-900">
            <tr>
              {header('symbol', 'symbol')}
              <th className="th">side</th>
              {header('r', 'R')}
              {header('pnl', 'PnL')}
              <th className="th">exit</th>
              {header('closed_at', 'closed')}
            </tr>
          </thead>
          <tbody>
            {filtered.map(t => (
              <tr key={t.id} className="transition-colors hover:bg-surface-2 dark:hover:bg-neutral-800/60">
                <td className="td">
                  <span className="mr-2 align-middle"><Dot tone={t.win ? 'success' : 'danger'} /></span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-200">{t.symbol.split('/')[0]}</span>
                  <span className="ml-1 text-2xs text-neutral-500">{t.timeframe}</span>
                </td>
                <td className="td text-2xs text-neutral-500">{t.side}</td>
                <td className={`td num font-medium ${t.r >= 0 ? 'text-success' : 'text-danger'}`}>
                  {fmtSigned(t.r)}R
                </td>
                <td className={`td num ${t.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                  {fmtSigned(t.pnl)}
                </td>
                <td className="td text-2xs text-neutral-500">{t.exit_reason}</td>
                <td className="td text-2xs text-neutral-500">{timeAgo(t.closed_at, now)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
