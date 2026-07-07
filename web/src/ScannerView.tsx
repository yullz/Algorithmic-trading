import { useEffect, useState } from 'react';
import { useErrors } from './api';
import { fmtPct, fmtPrice, fmtR, heatColor } from './format';
import { Chip, ConfBar, Dot, Empty, RegimeChip, Section, SideTag, Skeleton, Sparkline } from './ui';
import type { MarketTile, Plan, ScanResult } from './types';

export default function ScannerView({ scan, paused, onPause, onResume, onSelect }:
  { scan: ScanResult | null; paused: boolean;
    onPause: () => Promise<void>; onResume: () => Promise<void>;
    onSelect: (symbol: string, tf: string, plan?: Plan) => void }) {
  const [busy, setBusy] = useState(false);
  const errors = useErrors();
  const [toast, setToast] = useState<string | null>(null);

  const lastError = errors[0] ?? null;
  useEffect(() => {
    if (!lastError) return;
    setToast(lastError.message);
    const id = setTimeout(() => setToast(null), 6000);
    return () => clearTimeout(id);
  }, [lastError?.id]);

  const togglePause = async () => {
    setBusy(true);
    try {
      if (paused) await onResume(); else await onPause();
    } finally {
      setBusy(false);
    }
  };

  if (!scan) {
    return (
      <div className="space-y-5">
        <Section title="Ranked setups"><Skeleton rows={5} /></Section>
        <Section title="Market heatmap"><Skeleton rows={3} /></Section>
      </div>
    );
  }
  const plans = scan.plans ?? [];
  const market = scan.market ?? [];
  const suppressed = scan.suppressed_by_correlation ?? [];

  return (
    <div className="space-y-5">
      {toast && (
        <div className="rounded-lg border border-danger/30 bg-danger-dim px-4 py-2 text-sm text-danger">
          {toast}
        </div>
      )}

      <Section
        title={`Ranked setups — ${plans.length} of ${scan.candidates ?? 0} candidates`}
        right={
          <div className="flex items-center gap-3">
            <SymbolSearch market={market} onSelect={onSelect} />
            <span className="text-2xs text-neutral-500">
              {scan.pairs_scanned ?? 0} pairs · {(scan.duration_sec ?? 0).toFixed(0)}s
              {(scan.fetch_errors ?? 0) > 0 && ` · ${scan.fetch_errors} fetch errors`}
            </span>
            <button
              onClick={togglePause}
              disabled={busy}
              className={`rounded-md border px-2.5 py-1 text-2xs font-medium transition-colors ${
                paused
                  ? 'border-warning/30 bg-warning-dim text-warning hover:bg-warning/20'
                  : 'border-primary/30 bg-primary-dim text-primary hover:bg-primary/20'
              } disabled:opacity-50`}
            >
              {paused ? '▶ Resume scan' : '⏸ Pause scan'}
            </button>
          </div>
        }
      >
        {plans.length === 0 ? (
          <Empty
            title="No setups passed the gates."
            hint="Most bars have no edge — that's the system working, not failing."
          />
        ) : (
          <div className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-3">
            {plans.map((p, i) => (
              <SignalCard key={`${p.symbol}-${p.timeframe}`} plan={p} rank={i + 1} onSelect={onSelect} />
            ))}
          </div>
        )}
      </Section>

      <Section title={`Market heatmap — ${market.length} symbols (24h)`}>
        {market.length === 0 && (
          <Empty
            title="Waiting for the first scan…"
            hint="The background loop scans the whole universe on the configured cadence."
          />
        )}
        <div className="grid grid-cols-[repeat(auto-fill,minmax(92px,1fr))] gap-1.5 p-4">
          {market.map(m => (
            <button
              key={m.symbol}
              onClick={() => onSelect(m.symbol, '1h')}
              title={`${m.symbol} · ${fmtPrice(m.last)} · ${m.chg24h_pct >= 0 ? '+' : ''}${m.chg24h_pct}% 24h${m.picked ? ' · PICKED' : m.candidate ? ' · candidate' : ''}`}
              className={`group relative overflow-hidden rounded-lg px-2 py-2 text-left transition-all hover:scale-[1.03] hover:shadow-md ${
                m.picked ? 'ring-1 ring-primary' : m.candidate ? 'ring-1 ring-neutral-500/50' : ''
              }`}
              style={{ background: heatColor(m.chg24h_pct) }}
            >
              <div className="relative z-10">
                <div className="flex items-center justify-between">
                  <span className="truncate text-xs font-semibold text-neutral-100">{m.symbol.split('/')[0]}</span>
                  {m.picked && <Dot tone="primary" />}
                </div>
                <div className={`num mt-0.5 text-xs ${m.chg24h_pct >= 0 ? 'text-success' : 'text-danger'}`}>
                  {m.chg24h_pct >= 0 ? '+' : ''}{m.chg24h_pct.toFixed(1)}%
                </div>
                {m.spark && m.spark.length > 1 && (
                  <div className="mt-1 h-6 opacity-70">
                    <Sparkline
                      data={m.spark}
                      width={72}
                      height={20}
                      stroke={m.chg24h_pct >= 0 ? 'var(--success)' : 'var(--danger)'}
                    />
                  </div>
                )}
              </div>
            </button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-3 border-t border-line bg-surface-2/30 px-4 py-2 text-2xs text-neutral-500 dark:bg-neutral-800/20">
          <span className="inline-flex items-center gap-1.5"><Dot tone="primary" /> picked</span>
          <span className="inline-flex items-center gap-1.5"><span className="chip-dot bg-neutral-500/50" /> candidate</span>
          <span>tile color = 24h move · sparkline = 24h price</span>
        </div>
      </Section>

      {suppressed.length > 0 && (
        <Section title={`Suppressed by correlation — ${suppressed.length}`}>
          <div className="border-b border-line px-4 py-3 text-2xs text-neutral-500">
            These passed every gate but correlate &gt;80% with a higher-ranked pick —
            taking both would be the same bet twice.
          </div>
          <div className="grid gap-2 p-4 sm:grid-cols-2 lg:grid-cols-3">
            {suppressed.map(p => (
              <SuppressedCard key={`${p.symbol}-${p.timeframe}`} plan={p} onSelect={onSelect} />
            ))}
          </div>
        </Section>
      )}
    </div>
  );
}

function SignalCard({ plan, rank, onSelect }:
  { plan: Plan; rank: number; onSelect: (symbol: string, tf: string, plan?: Plan) => void }) {
  const quote = plan.symbol.split('/')[1]?.split(':')[0];
  return (
    <button
      onClick={() => onSelect(plan.symbol, plan.timeframe, plan)}
      className="card group flex flex-col gap-3 p-4 text-left transition-all hover:border-primary/30 hover:bg-surface-2 dark:hover:bg-neutral-800/80"
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="num text-xs text-neutral-500">#{rank}</span>
            <span className="text-base font-semibold text-neutral-900 dark:text-neutral-100">
              {plan.symbol.split('/')[0]}
            </span>
            {quote && <span className="text-2xs text-neutral-500">/{quote}</span>}
          </div>
          <div className="mt-1 flex items-center gap-2">
            <SideTag side={plan.side} />
            <Chip>{plan.timeframe}</Chip>
            <RegimeChip regime={plan.regime} />
          </div>
        </div>
        <ConfBar value={plan.confidence} />
      </div>

      <div className="grid grid-cols-3 gap-2 rounded-lg bg-surface-2 p-2 dark:bg-neutral-800/60">
        <Metric label="EV" value={fmtR(plan.expected_value_r)}
          tone={plan.expected_value_r >= 0 ? 'success' : 'danger'} />
        <Metric label="win%" value={fmtPct(plan.expected_win_rate, 0)} />
        <Metric label="R:R" value={plan.reward_risk.toFixed(2)} />
      </div>

      <div className="flex flex-wrap gap-1.5">
        {plan.families.slice(0, 3).map(f => (
          <Chip key={f}>{f.replace(/_/g, ' ')}</Chip>
        ))}
        {plan.families.length > 3 && <Chip>+{plan.families.length - 3}</Chip>}
        {plan.warnings.length > 0 && (
          <Chip tone="warning" title={plan.warnings.join('\n')}>⚠ {plan.warnings.length}</Chip>
        )}
      </div>
    </button>
  );
}

function SuppressedCard({ plan, onSelect }:
  { plan: Plan; onSelect: (symbol: string, tf: string, plan?: Plan) => void }) {
  return (
    <button
      onClick={() => onSelect(plan.symbol, plan.timeframe, plan)}
      className="card flex items-center justify-between gap-3 p-3 text-left transition-colors hover:bg-surface-2 dark:hover:bg-neutral-800/80"
      title={`suppressed by ${plan.suppressed_by}`}
    >
      <div>
        <div className="flex items-center gap-2">
          <span className="font-medium text-neutral-900 dark:text-neutral-100">{plan.symbol.split('/')[0]}</span>
          <SideTag side={plan.side} />
          <span className="text-2xs text-neutral-500">{plan.timeframe}</span>
        </div>
        <div className="mt-1 text-2xs text-neutral-500">
          EV {fmtR(plan.expected_value_r)} · conf {Math.round(plan.confidence * 100)}%
        </div>
      </div>
      <div className="text-right">
        <div className="text-2xs text-neutral-500">suppressed by</div>
        <div className="text-xs font-medium text-neutral-300">{plan.suppressed_by?.split('/')[0]}</div>
      </div>
    </button>
  );
}

function SymbolSearch({ market, onSelect }:
  { market: MarketTile[]; onSelect: (symbol: string, tf: string) => void }) {
  const [q, setQ] = useState('');
  const submit = () => {
    const term = q.trim().toUpperCase();
    if (!term) return;
    const hit = market.find(m => m.symbol.toUpperCase() === term)
      || market.find(m => m.symbol.split('/')[0].toUpperCase() === term)
      || market.find(m => m.symbol.toUpperCase().startsWith(term));
    if (hit) { onSelect(hit.symbol, '1h'); setQ(''); }
  };
  return (
    <div className="relative">
      <input
        list="symbol-list"
        value={q}
        onChange={e => setQ(e.target.value)}
        onKeyDown={e => { if (e.key === 'Enter') submit(); }}
        placeholder="chart any symbol…"
        className="w-36 rounded-md border border-line bg-surface-2 px-2 py-1 text-2xs text-neutral-700 placeholder:text-neutral-500 focus:outline-none focus:ring-1 focus:ring-primary dark:bg-neutral-800 dark:text-neutral-200"
      />
      <datalist id="symbol-list">
        {market.map(m => <option key={m.symbol} value={m.symbol.split('/')[0]} />)}
      </datalist>
    </div>
  );
}

function Metric({ label, value, tone }:
  { label: string; value: string; tone?: 'success' | 'danger' | 'warning' }) {
  const color = tone === 'success' ? 'text-success'
    : tone === 'danger' ? 'text-danger'
    : tone === 'warning' ? 'text-warning'
    : 'text-neutral-900 dark:text-neutral-100';
  return (
    <div className="text-center">
      <div className="text-2xs text-neutral-500">{label}</div>
      <div className={`num text-sm font-semibold ${color}`}>{value}</div>
    </div>
  );
}
