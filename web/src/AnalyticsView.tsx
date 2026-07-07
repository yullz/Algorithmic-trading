import { useEffect, useState } from 'react';
import { get } from './api';
import { fmtPct } from './format';
import { Chip, Dot, Empty, Section, Sparkline, Stat } from './ui';
import type { BacktestReport, MLModelInfo, WalkforwardReport } from './types';

export default function AnalyticsView() {
  const [bt, setBt] = useState<BacktestReport | null>(null);
  const [wf, setWf] = useState<WalkforwardReport | null>(null);
  const [cal, setCal] = useState<Record<string, number> | null>(null);
  const [ml, setMl] = useState<MLModelInfo | null>(null);

  useEffect(() => {
    get<BacktestReport>('/api/backtest').then(setBt).catch(() => {});
    get<WalkforwardReport>('/api/walkforward').then(setWf).catch(() => {});
    get<Record<string, number>>('/api/calibration').then(setCal).catch(() => {});
    get<MLModelInfo>('/api/mlmodel').then(setMl).catch(() => {});
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
                  <thead className="bg-surface-2/50 dark:bg-slate-800/40">
                    <tr>
                      <th className="th">metric</th>
                      <th className="th text-right">in-sample</th>
                      <th className="th text-right">out-of-sample</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(['trades', 'win_rate', 'expectancy_r', 'profit_factor', 'max_drawdown_r'] as const).map(k => (
                      <tr key={k} className="hover:bg-surface-2 dark:hover:bg-slate-800/30">
                        <td className="td text-slate-500">{k.replace(/_/g, ' ')}</td>
                        <td className="td num text-right">{fmtMetric(k, wf.in_sample?.[k])}</td>
                        <td className="td num text-right font-medium text-slate-900 dark:text-slate-100">
                          {fmtMetric(k, wf.out_of_sample?.[k])}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {wf.oos_equity_curve && wf.oos_equity_curve.length > 1 && (
                <div className="card flex min-w-[220px] flex-col items-center justify-center gap-2 p-4">
                  <span className="text-2xs font-semibold uppercase tracking-wider text-slate-500">OOS equity (R)</span>
                  <Sparkline data={wf.oos_equity_curve} width={200} height={64} fill="var(--secondary)" stroke="var(--secondary)" />
                </div>
              )}
            </div>
          </div>
        )}
      </Section>

      <div className="grid gap-4 lg:grid-cols-2">
        <Section title={`Factor win rates — ${factors.length} calibrated`}>
          {factors.length === 0 ? (
            <Empty title="Uncalibrated" hint="Factor hit rates appear after a backtest run." />
          ) : (
            <div className="max-h-[28rem] space-y-2 overflow-y-auto p-4">
              {factors.map(([name, rate]) => (
                <div key={name} className="flex items-center gap-3">
                  <span className="num w-48 truncate text-2xs text-slate-500" title={name}>{name}</span>
                  <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-2 dark:bg-slate-800">
                    <div className="absolute inset-y-0 left-1/2 w-px bg-slate-500/40" title="50%" />
                    <div
                      className={`h-full rounded-full ${rate >= 0.5 ? 'bg-success/70' : 'bg-danger/70'}`}
                      style={{ width: `${Math.min(rate * 100, 100)}%` }}
                    />
                  </div>
                  <span className="num w-10 text-right text-2xs text-slate-700 dark:text-slate-300">{fmtPct(rate, 0)}</span>
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
                <span className="text-2xs text-slate-500">trained {ml.trained_at?.slice(0, 16)}</span>
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
              <p className="text-2xs leading-relaxed text-slate-500">
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
