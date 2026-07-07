import { useState } from 'react';
import { useHistorySignals, useHistorySummary, useHistoryTrades } from './api';
import { fmtPct, fmtPrice, fmtR, fmtSigned, timeAgo } from './format';
import { Empty, RegimeChip, Section, SideTag, Skeleton, Stat } from './ui';
import type { HistoryBreakdown } from './types';

export default function HistoryView() {
  const [signalFilters, setSignalFilters] = useState({
    symbol: '', timeframe: '', side: '', regime: '', from: '', to: '', limit: 100,
  });
  const [tradeFilters, setTradeFilters] = useState({
    symbol: '', outcome: '', from: '', to: '', limit: 100,
  });

  const signals = useHistorySignals(signalFilters);
  const trades = useHistoryTrades(tradeFilters);
  const summary = useHistorySummary();

  const closed = trades?.filter(t => t.outcome && t.outcome !== 'open') ?? [];
  const wins = closed.filter(t => (t.realized_r ?? 0) > 0);
  const winRate = closed.length ? wins.length / closed.length : 0;
  const avgR = closed.length
    ? closed.reduce((s, t) => s + (t.realized_r ?? 0), 0) / closed.length
    : 0;
  const profitFactor = (() => {
    const gains = closed.filter(t => (t.realized_r ?? 0) > 0).reduce((s, t) => s + (t.realized_r ?? 0), 0);
    const losses = Math.abs(closed.filter(t => (t.realized_r ?? 0) < 0).reduce((s, t) => s + (t.realized_r ?? 0), 0));
    return losses ? gains / losses : gains ? Infinity : 0;
  })();

  return (
    <div className="space-y-5">
      <Section title="Historical summary">
        {!summary ? (
          <Skeleton rows={2} />
        ) : (
          <div className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-5">
            <Stat label="total trades" value={summary.total_trades} />
            <Stat label="closed trades" value={summary.closed_trades} />
            <Stat label="win rate" value={fmtPct(summary.win_rate ?? winRate, 1)}
              tone={(summary.win_rate ?? winRate) >= 0.5 ? 'success' : 'danger'} />
            <Stat label="avg realized R" value={fmtR(summary.avg_realized_r ?? avgR)}
              tone={(summary.avg_realized_r ?? avgR) >= 0 ? 'success' : 'danger'} />
            <Stat label="profit factor" value={profitFactor === Infinity ? '∞' : profitFactor.toFixed(2)}
              tone={profitFactor >= 1 ? 'success' : 'danger'} />
          </div>
        )}
      </Section>

      <div className="grid gap-5 lg:grid-cols-2">
        <BreakdownCard title="Win rate by regime" rows={summary?.by_regime ?? []} />
        <BreakdownCard title="Win rate by setup kind" rows={summary?.by_setup_kind ?? []} />
      </div>

      <Section title="Signal history">
        <div className="space-y-3 p-4">
          <div className="flex flex-wrap gap-2">
            <FilterInput placeholder="symbol" value={signalFilters.symbol}
              onChange={v => setSignalFilters(f => ({ ...f, symbol: v }))} />
            <FilterInput placeholder="timeframe" value={signalFilters.timeframe}
              onChange={v => setSignalFilters(f => ({ ...f, timeframe: v }))} />
            <FilterSelect value={signalFilters.side}
              onChange={v => setSignalFilters(f => ({ ...f, side: v }))}>
              <option value="">all sides</option>
              <option value="LONG">LONG</option>
              <option value="SHORT">SHORT</option>
            </FilterSelect>
            <FilterInput placeholder="regime" value={signalFilters.regime}
              onChange={v => setSignalFilters(f => ({ ...f, regime: v }))} />
            <FilterInput type="date" label="from" value={signalFilters.from}
              onChange={v => setSignalFilters(f => ({ ...f, from: v }))} />
            <FilterInput type="date" label="to" value={signalFilters.to}
              onChange={v => setSignalFilters(f => ({ ...f, to: v }))} />
          </div>

          {!signals ? <Skeleton rows={4} /> : signals.length === 0 ? <Empty title="No signals match" /> : (
            <div className="table-responsive max-h-96 rounded-lg border border-line">
              <table className="w-full whitespace-nowrap text-sm">
                <thead className="sticky top-0 z-10 bg-surface-1 dark:bg-neutral-900">
                  <tr>
                    <th className="th">symbol</th>
                    <th className="th">side</th>
                    <th className="th">timeframe</th>
                    <th className="th">kind</th>
                    <th className="th">regime</th>
                    <th className="th">entry</th>
                    <th className="th">stop</th>
                    <th className="th">EV</th>
                    <th className="th">timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {signals.map(s => (
                    <tr key={s.id} className="transition-colors hover:bg-surface-2 dark:hover:bg-neutral-800/60">
                      <td className="td font-medium text-neutral-900 dark:text-neutral-200">{s.symbol.split('/')[0]}</td>
                      <td className="td"><SideTag side={s.side as 'LONG' | 'SHORT'} /></td>
                      <td className="td text-2xs text-neutral-500">{s.timeframe}</td>
                      <td className="td text-2xs text-neutral-500">{s.kind}</td>
                      <td className="td"><RegimeChip regime={s.btc_regime ?? ''} /></td>
                      <td className="td num text-2xs">{fmtPrice(s.entry)}</td>
                      <td className="td num text-2xs">{fmtPrice(s.stop)}</td>
                      <td className={`td num text-2xs font-medium ${(s.expected_value_r ?? 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                        {fmtR(s.expected_value_r)}
                      </td>
                      <td className="td text-2xs text-neutral-500">{timeAgo(s.timestamp)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Section>

      <Section title="Trade history">
        <div className="space-y-3 p-4">
          <div className="flex flex-wrap gap-2">
            <FilterInput placeholder="symbol" value={tradeFilters.symbol}
              onChange={v => setTradeFilters(f => ({ ...f, symbol: v }))} />
            <FilterInput placeholder="outcome" value={tradeFilters.outcome}
              onChange={v => setTradeFilters(f => ({ ...f, outcome: v }))} />
            <FilterInput type="date" label="from" value={tradeFilters.from}
              onChange={v => setTradeFilters(f => ({ ...f, from: v }))} />
            <FilterInput type="date" label="to" value={tradeFilters.to}
              onChange={v => setTradeFilters(f => ({ ...f, to: v }))} />
          </div>

          {!trades ? <Skeleton rows={4} /> : trades.length === 0 ? <Empty title="No trades match" /> : (
            <div className="table-responsive max-h-96 rounded-lg border border-line">
              <table className="w-full whitespace-nowrap text-sm">
                <thead className="sticky top-0 z-10 bg-surface-1 dark:bg-neutral-900">
                  <tr>
                    <th className="th">symbol</th>
                    <th className="th">side</th>
                    <th className="th">status</th>
                    <th className="th">outcome</th>
                    <th className="th">realized R</th>
                    <th className="th">mtm PnL</th>
                    <th className="th">opened</th>
                    <th className="th">closed</th>
                  </tr>
                </thead>
                <tbody>
                  {trades.map(t => (
                    <tr key={t.id} className="transition-colors hover:bg-surface-2 dark:hover:bg-neutral-800/60">
                      <td className="td font-medium text-neutral-900 dark:text-neutral-200">{t.symbol.split('/')[0]}</td>
                      <td className="td"><SideTag side={t.side as 'LONG' | 'SHORT'} /></td>
                      <td className="td text-2xs text-neutral-500">{t.status}</td>
                      <td className="td text-2xs text-neutral-500">{t.outcome ?? '—'}</td>
                      <td className={`td num text-2xs font-medium ${(t.realized_r ?? 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                        {fmtR(t.realized_r)}
                      </td>
                      <td className={`td num text-2xs ${(t.mtm_pnl ?? 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                        {fmtSigned(t.mtm_pnl ?? 0)}
                      </td>
                      <td className="td text-2xs text-neutral-500">{timeAgo(t.timestamp)}</td>
                      <td className="td text-2xs text-neutral-500">{t.closed_at ? timeAgo(t.closed_at) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </Section>
    </div>
  );
}

function FilterInput({ placeholder, value, onChange, type = 'text', label }:
  { placeholder?: string; value: string; onChange: (v: string) => void; type?: string; label?: string }) {
  return (
    <label className="inline-flex items-center gap-1.5 text-2xs text-neutral-500">
      {label}
      <input
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="rounded-md border border-line bg-surface-1 px-2 py-1 text-xs text-neutral-700 outline-none focus:border-primary dark:bg-neutral-900 dark:text-neutral-300"
      />
    </label>
  );
}

function FilterSelect({ value, onChange, children }:
  { value: string; onChange: (v: string) => void; children: React.ReactNode }) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      className="rounded-md border border-line bg-surface-1 px-2 py-1 text-xs text-neutral-700 outline-none focus:border-primary dark:bg-neutral-900 dark:text-neutral-300"
    >
      {children}
    </select>
  );
}

function BreakdownCard({ title, rows }: { title: string; rows: HistoryBreakdown[] }) {
  return (
    <Section title={title}>
      {rows.length === 0 ? (
        <Empty title="No data" />
      ) : (
        <div className="max-h-60 space-y-2 overflow-y-auto p-4">
          {rows.map(r => {
            const label = r.btc_regime ?? r.kind ?? r.month ?? 'unknown';
            const rate = r.trades ? r.wins / r.trades : 0;
            return (
              <div key={label} className="flex items-center gap-3">
                <span className="w-32 truncate text-2xs text-neutral-500" title={label}>{label}</span>
                <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2 dark:bg-neutral-800">
                  <div className="absolute inset-y-0 left-1/2 w-px bg-neutral-500/40" title="50%" />
                  <div
                    className={`h-full rounded-full ${rate >= 0.5 ? 'bg-success/70' : 'bg-danger/70'}`}
                    style={{ width: `${Math.min(rate * 100, 100)}%` }}
                  />
                </div>
                <span className="num w-10 text-right text-2xs text-neutral-700 dark:text-neutral-300">{fmtPct(rate, 0)}</span>
                <span className="num w-10 text-right text-2xs text-neutral-500">{r.trades}</span>
              </div>
            );
          })}
        </div>
      )}
    </Section>
  );
}
