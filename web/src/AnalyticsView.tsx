import { useEffect, useState } from 'react';
import { get } from './api';
import { fmtPct } from './format';
import { Chip, Dot, Empty, Section, Sparkline, Stat } from './ui';
import type { BacktestReport, MLModelInfo, ReliabilityReport, WalkforwardReport } from './types';

export default function AnalyticsView() {
  const [bt, setBt] = useState<BacktestReport | null>(null);
  const [wf, setWf] = useState<WalkforwardReport | null>(null);
  const [cal, setCal] = useState<Record<string, number> | null>(null);
  const [ml, setMl] = useState<MLModelInfo | null>(null);
  const [rel, setRel] = useState<ReliabilityReport | null>(null);

  useEffect(() => {
    get<BacktestReport>('/api/backtest').then(setBt).catch(() => {});
    get<WalkforwardReport>('/api/walkforward').then(setWf).catch(() => {});
    get<Record<string, number>>('/api/calibration').then(setCal).catch(() => {});
    get<MLModelInfo>('/api/mlmodel').then(setMl).catch(() => {});
    get<ReliabilityReport>('/api/analytics/reliability').then(setRel).catch(() => {});
  }, []);

  const s = bt?.summary ?? {};
  const factors = Object.entries(cal ?? {})
    .filter(([k]) => !k.startsWith('_') && !k.includes('|'))
    .sort((a, b) => b[1] - a[1]);
  const verdictHolds = wf?.verdict?.includes('holds') && !wf.verdict.includes('NOT');

  return (
    <div className="space-y-5">
      <Section title={`Backtest (in-sample${bt?.source ? ` · ${bt.source}` : ''})`}>
        {!bt?.n_trades ? (
          <Empty title="No backtest report yet"
            hint="Run: python backtest.py --deep --limit 5000 --export-dataset --walkforward" />
        ) : (
          <div className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-7">
            <div className="col-span-1 lg:col-span-5">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-5">
                <Stat label="trades" value={s.trades ?? 0} />
                <Stat label="win rate" value={fmtPct(s.win_rate ?? 0, 0)} />
                <Stat label="expectancy" value={`${(s.expectancy_r ?? 0).toFixed(2)}R`}
                  tone={(s.expectancy_r ?? 0) >= 0 ? 'success' : 'danger'} />
                <Stat label="profit factor" value={(s.profit_factor ?? 0).toFixed(2)}
                  tone={(s.profit_factor ?? 0) >= 1 ? 'success' : 'danger'} />
                <Stat label="max drawdown" value={`${(s.max_drawdown_r ?? 0).toFixed(1)}R`} tone="warning" />
              </div>
            </div>
            <div className="card col-span-1 flex items-center justify-center p-3 lg:col-span-2">
              <Sparkline data={bt.equity_curve} width={220} height={56} fill="var(--primary)" />
            </div>
          </div>
        )}
      </Section>

      <Section title="Walk-forward (the honest test)">
        {!wf?.out_of_sample ? (
          <Empty title="No walk-forward report yet"
            hint="Run: python backtest.py --walkforward — out-of-sample validation is what separates edge from overfit." />
        ) : (
          <div className="p-4">
            <div className={`mb-4 inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm font-semibold ${
              verdictHolds
                ? 'border-success/30 bg-success-dim text-success'
                : 'border-danger/30 bg-danger-dim text-danger'}`}>
              <Dot tone={verdictHolds ? 'success' : 'danger'} />
              {wf.verdict}
            </div>
            <div className="grid gap-4 lg:grid-cols-[1fr_auto]">
              <div className="table-responsive rounded-lg border border-line">
                <table className="w-full text-sm">
                  <thead className="bg-surface-2/50 dark:bg-neutral-800/40">
                    <tr>
                      <th className="th">metric</th>
                      <th className="th text-right">in-sample</th>
                      <th className="th text-right">out-of-sample</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(['trades', 'win_rate', 'expectancy_r', 'profit_factor', 'max_drawdown_r'] as const).map(k => (
                      <tr key={k} className="hover:bg-surface-2 dark:hover:bg-neutral-800/30">
                        <td className="td text-neutral-500">{k.replace(/_/g, ' ')}</td>
                        <td className="td num text-right">{fmtMetric(k, wf.in_sample?.[k])}</td>
                        <td className="td num text-right font-medium text-neutral-900 dark:text-neutral-100">
                          {fmtMetric(k, wf.out_of_sample?.[k])}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {wf.oos_equity_curve && wf.oos_equity_curve.length > 1 && (
                <div className="card flex min-w-[220px] flex-col items-center justify-center gap-2 p-4">
                  <span className="text-2xs font-semibold uppercase tracking-wider text-neutral-500">OOS equity (R)</span>
                  <Sparkline data={wf.oos_equity_curve} width={200} height={64} fill="var(--secondary)" stroke="var(--secondary)" />
                </div>
              )}
            </div>
          </div>
        )}
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title="Calibration reliability (predicted vs realized)">
          {!rel?.present || !rel.buckets?.length ? (
            <Empty title="No reliability data"
              hint="Run backtest.py --export-dataset — this compares predicted win rate to realized outcomes." />
          ) : (
            <div className="p-4">
              <div className="mx-auto max-w-sm"><ReliabilityPlot buckets={rel.buckets} /></div>
              <p className="mt-2 text-center text-2xs text-neutral-500">
                {rel.n?.toLocaleString()} trades · points on the diagonal = well-calibrated
              </p>
            </div>
          )}
        </Section>

        <Section title="ML feature importance">
          {!ml?.top_features?.length ? (
            <Empty title="No trained model"
              hint="Train the meta-model to see which features drive its predictions." />
          ) : (
            <div className="space-y-1.5 p-4">
              {ml.top_features.map(f => {
                const max = ml.top_features![0].importance || 1;
                return (
                  <div key={f.name} className="flex items-center gap-2">
                    <span className="num w-40 truncate text-2xs text-neutral-500" title={f.name}>{f.name}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2 dark:bg-neutral-800">
                      <div className="h-full rounded-full bg-secondary/70"
                        style={{ width: `${Math.max(2, Math.min((f.importance / max) * 100, 100))}%` }} />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </Section>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title={`Factor win rates — ${factors.length} calibrated`}>
          {factors.length === 0 ? (
            <Empty title="Uncalibrated" hint="Factor hit rates appear after a backtest run." />
          ) : (
            <div className="max-h-[28rem] space-y-2 overflow-y-auto p-4">
              {factors.map(([name, rate]) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="num w-48 truncate text-2xs text-neutral-500" title={name}>{name}</span>
                  <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2 dark:bg-neutral-800">
                    <div className="absolute inset-y-0 left-1/2 w-px bg-neutral-500/40" title="50%" />
                    <div
                      className={`h-full rounded-full ${rate >= 0.5 ? 'bg-success/70' : 'bg-danger/70'}`}
                      style={{ width: `${Math.min(rate * 100, 100)}%` }}
                    />
                  </div>
                  <span className="num w-10 text-right text-2xs text-neutral-700 dark:text-neutral-300">{fmtPct(rate, 0)}</span>
                </div>
              ))}
            </div>
          )}
        </Section>

        <Section title="ML meta-model">
          {!ml?.present ? (
            <Empty title="No trained model"
              hint="python backtest.py --export-dataset  →  python -m algotrader.ml.train" />
          ) : (
            <div className="space-y-4 p-4">
              <div className="flex items-center gap-2">
                <Chip tone={ml.trusted ? 'success' : 'warning'}>
                  {ml.trusted ? 'TRUSTED — in the blend' : 'NOT TRUSTED — rules only'}
                </Chip>
                <span className="text-2xs text-neutral-500">trained {ml.trained_at?.slice(0, 16)}</span>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Stat label="OOS AUC" value={(ml.auc_valid ?? 0).toFixed(3)}
                  tone={(ml.auc_valid ?? 0) > 0.53 ? 'success' : 'warning'}
                  sub="needs > 0.53" />
                <Stat label="training trades" value={ml.n_train ?? 0}
                  tone={(ml.n_train ?? 0) >= 300 ? undefined : 'warning'} sub="needs ≥ 300" />
                <Stat label="Brier score" value={(ml.brier_valid ?? 0).toFixed(3)} sub="lower is better" />
                <Stat label="base win rate" value={fmtPct(ml.base_rate ?? 0, 0)} sub="class balance" />
              </div>
              <p className="text-2xs leading-relaxed text-neutral-500">
                The model re-scores rule-approved setups only. Its blend weight is
                earned from out-of-sample skill and capped at 50% — and the final
                estimate always stays inside the 30–78% honesty cap.
              </p>
            </div>
          )}
        </Section>
      </div>
    </div>
  );
}

function fmtMetric(k: string, v: number | undefined): string {
  if (v === undefined || v === null) return '—';
  if (k === 'win_rate') return fmtPct(v, 0);
  if (k === 'trades') return String(v);
  return v.toFixed(2);
}

function ReliabilityPlot({ buckets }:
  { buckets: { predicted: number; realized: number; n: number }[] }) {
  const W = 280, H = 220, pad = 30;
  const x = (v: number) => pad + v * (W - pad * 2);
  const y = (v: number) => H - pad - v * (H - pad * 2);
  const maxN = Math.max(...buckets.map(b => b.n), 1);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
      {[0, 0.25, 0.5, 0.75, 1].map(t => (
        <g key={t}>
          <line x1={pad} y1={y(t)} x2={W - pad} y2={y(t)} stroke="var(--line)" strokeWidth="1" />
          <text x={pad - 4} y={y(t) + 3} textAnchor="end" fontSize="7" fill="#64748b">
            {Math.round(t * 100)}
          </text>
          <text x={x(t)} y={H - pad + 12} textAnchor="middle" fontSize="7" fill="#64748b">
            {Math.round(t * 100)}
          </text>
        </g>
      ))}
      {/* perfect-calibration diagonal */}
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)}
        stroke="var(--line-strong)" strokeDasharray="4 3" strokeWidth="1" />
      {/* observed reliability curve */}
      <polyline fill="none" stroke="var(--primary)" strokeWidth="1.5"
        points={buckets.map(b => `${x(b.predicted)},${y(b.realized)}`).join(' ')} />
      {buckets.map((b, i) => (
        <circle key={i} cx={x(b.predicted)} cy={y(b.realized)}
          r={3 + 4 * (b.n / maxN)}
          fill={Math.abs(b.predicted - b.realized) < 0.06 ? 'var(--success)' : 'var(--warning)'}
          fillOpacity="0.85">
          <title>{`predicted ${(b.predicted * 100).toFixed(0)}% → realized `
            + `${(b.realized * 100).toFixed(0)}% (n=${b.n})`}</title>
        </circle>
      ))}
      <text x={W / 2} y={H - 4} textAnchor="middle" fontSize="8" fill="#64748b">predicted win %</text>
      <text x={9} y={H / 2} textAnchor="middle" fontSize="8" fill="#64748b"
        transform={`rotate(-90 9 ${H / 2})`}>realized win %</text>
    </svg>
  );
}
