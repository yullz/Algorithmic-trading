import { useTick } from './api';
import { fmtMoney, fmtPct, secondsSince, timeAgo } from './format';
import { Chip, Dot, RegimeChip } from './ui';
import type { Config, Health, Positions, ScanResult } from './types';

export default function StatusBar({ cfg, scan, positions, connected, health, errorCount, onToggleErrors }:
  { cfg: Config | null; scan: ScanResult | null; positions: Positions | null;
    connected: boolean; health: Health | null; errorCount: number;
    onToggleErrors: () => void }) {
  const now = useTick(5000);
  const interval = cfg?.scan_interval_sec ?? 300;
  const age = secondsSince(scan?.scanned_at ?? null, now);
  const freshness = Math.min(age / interval, 1);
  const scanTone = age > interval * 5 ? 'danger'
    : age > interval * 2 ? 'warning'
    : 'success';

  const mode = !cfg ? '…'
    : cfg.execution_mode !== 'live' || !cfg.live_attached ? 'PAPER'
    : cfg.testnet ? 'LIVE · TESTNET' : 'LIVE · MAINNET';
  const modeTone: 'success' | 'warning' | 'danger' = mode === 'PAPER' ? 'success'
    : mode === 'LIVE · TESTNET' ? 'warning' : 'danger';

  const dayPnl = positions
    ? positions.mtm_equity - (positions.day_anchor?.equity ?? positions.mtm_equity)
    : 0;

  return (
    <div className="sticky top-0 z-20 border-b border-line bg-surface-1/90 backdrop-blur dark:bg-slate-900/90">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2">
        <div className="flex items-center gap-2">
          <Dot tone={connected ? 'success' : 'danger'} pulse={!connected} />
          <span className="text-sm font-bold tracking-tight text-slate-900 dark:text-slate-100">Edge Terminal</span>
        </div>

        <Chip>
          {cfg ? `${cfg.exchange} · ${cfg.universe_mode === 'top_volume'
            ? `top ${cfg.universe_size}` : 'static'}` : '…'}
        </Chip>

        <div className="flex items-center gap-1.5">
          <span className="text-2xs font-semibold uppercase tracking-wider text-slate-500">BTC</span>
          <RegimeChip regime={scan?.btc_regime ?? ''} />
        </div>

        <Chip tone={modeTone === 'success' ? 'success' : modeTone === 'warning' ? 'warning' : 'danger'}>{mode}</Chip>

        {cfg && !cfg.calibrated && (
          <Chip tone="warning" title="run: python backtest.py">UNCALIBRATED</Chip>
        )}

        <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-1.5">
          {positions && (
            <>
              <span className="num text-sm font-semibold text-slate-900 dark:text-slate-200">
                {fmtMoney(positions.mtm_equity)} <span className="text-2xs font-normal text-slate-500">USDT</span>
              </span>
              <span className={`num text-sm font-semibold ${dayPnl >= 0 ? 'text-success' : 'text-danger'}`}>
                {dayPnl >= 0 ? '+' : ''}{fmtMoney(dayPnl)} <span className="text-2xs font-normal text-slate-500">today</span>
              </span>
            </>
          )}

          <div className="flex items-center gap-2">
            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-2 dark:bg-slate-800" title={`scan age / interval`}>
              <div
                className={`h-full rounded-full ${
                  scanTone === 'success' ? 'bg-success'
                  : scanTone === 'warning' ? 'bg-warning'
                  : 'bg-danger'
                }`}
                style={{ width: `${freshness * 100}%` }}
              />
            </div>
            <Chip tone={scanTone === 'success' ? 'success' : scanTone === 'warning' ? 'warning' : 'danger'}
              title={scan?.scanned_at ?? 'no scan yet'}>
              {age > interval * 5 ? 'STALE · ' : ''}scan {timeAgo(scan?.scanned_at ?? null, now)}
            </Chip>
          </div>

          {!connected && <Chip tone="warning"><Dot tone="warning" pulse /> reconnecting…</Chip>}

          <button
            onClick={onToggleErrors}
            className="relative rounded-md p-1.5 text-slate-500 transition-colors hover:bg-surface-2 hover:text-slate-700 dark:hover:text-slate-300"
            aria-label="Error log"
            title="Error log"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0"/></svg>
            {errorCount > 0 && (
              <span className="absolute -right-1 -top-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-danger px-1 text-[10px] font-bold text-white">
                {errorCount > 99 ? '99+' : errorCount}
              </span>
            )}
          </button>
        </div>
      </div>

      <SystemHealth health={health} />
    </div>
  );
}

function SystemHealth({ health }: { health: Health | null }) {
  if (!health) {
    return (
      <div className="border-t border-line bg-surface-1/70 px-4 py-1.5 dark:bg-slate-900/70">
        <div className="mx-auto flex max-w-7xl items-center gap-3 text-2xs text-slate-500">
          System health loading…
        </div>
      </div>
    );
  }

  const age = health.last_scan_age_sec;
  const ageTone = age === null ? 'danger'
    : age > 600 ? 'danger'
    : age > 120 ? 'warning'
    : 'success';

  const cb = health.circuit_breakers;
  const anyBreaker = cb.kill_switch || cb.daily_loss_triggered || cb.losing_streak_triggered;

  return (
    <div className="border-t border-line bg-surface-1/70 px-4 py-1.5 dark:bg-slate-900/70">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-4 gap-y-1 text-2xs">
        <HealthChip
          tone={ageTone}
          label="last scan"
          value={age === null ? 'unknown' : `${age}s ago`}
        />
        <HealthChip
          tone={health.data_fresh ? 'success' : 'warning'}
          label="data freshness"
          value={health.data_fresh ? 'fresh' : 'stale'}
        />
        <HealthChip
          tone={health.calibration_stale ? 'warning' : 'success'}
          label="calibration"
          value={health.calibration_stale ? 'stale' : 'ok'}
        />
        <HealthChip
          tone={health.model_trusted ? 'success' : 'warning'}
          label="model"
          value={health.model_trusted ? 'trusted' : 'untrusted'}
        />
        <HealthChip
          tone={anyBreaker ? 'danger' : 'success'}
          label="circuit breakers"
          value={anyBreaker ? 'TRIPPED' : 'ok'}
        />

        <div className="ml-auto flex items-center gap-3 text-slate-500">
          {cb.kill_switch && <span className="font-medium text-danger">KILL SWITCH</span>}
          <span>daily DD {fmtPct(cb.daily_loss_pct / 100, 2)}</span>
          <span>streak {cb.consecutive_losses}</span>
          {health.paused && <span className="font-medium text-warning">SCAN PAUSED</span>}
        </div>
      </div>
    </div>
  );
}

function HealthChip({ tone, label, value }:
  { tone: 'success' | 'warning' | 'danger'; label: string; value: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <Dot tone={tone} />
      <span className="text-slate-500">{label}:</span>
      <span className={`font-medium ${
        tone === 'success' ? 'text-success'
        : tone === 'warning' ? 'text-warning'
        : 'text-danger'
      }`}>{value}</span>
    </div>
  );
}
