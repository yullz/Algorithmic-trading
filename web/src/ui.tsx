import type { ReactNode } from 'react';

type Tone = 'neutral' | 'accent' | 'primary' | 'secondary' | 'success' | 'danger' | 'warning' | 'long' | 'short' | 'warn';

const TONE: Record<Tone, string> = {
  neutral: 'bg-surface-2 text-slate-600 border-line dark:bg-slate-800 dark:text-slate-300 dark:border-slate-700',
  accent: 'bg-primary-dim text-primary border-primary/30',
  primary: 'bg-primary-dim text-primary border-primary/30',
  secondary: 'bg-secondary-dim text-secondary border-secondary/30',
  success: 'bg-success-dim text-success border-success/30',
  long: 'bg-success-dim text-success border-success/30',
  danger: 'bg-danger-dim text-danger border-danger/30',
  short: 'bg-danger-dim text-danger border-danger/30',
  warning: 'bg-warning-dim text-warning border-warning/30',
  warn: 'bg-warning-dim text-warning border-warning/30',
};

export function Chip({ tone = 'neutral', children, title }:
  { tone?: Tone; children: ReactNode; title?: string }) {
  return (
    <span title={title}
      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-2xs font-medium ${TONE[tone]}`}>
      {children}
    </span>
  );
}

export function SideTag({ side }: { side: 'LONG' | 'SHORT' }) {
  return <Chip tone={side === 'LONG' ? 'success' : 'danger'}>{side}</Chip>;
}

export function RegimeChip({ regime }: { regime: string }) {
  const tone: Tone = regime === 'trend_up' ? 'success'
    : regime === 'trend_down' ? 'danger'
    : regime === 'volatile' ? 'warning' : 'neutral';
  return <Chip tone={tone}>{regime.replace(/_/g, ' ') || '—'}</Chip>;
}

export function ConfBar({ value }: { value: number }) {
  const pct = Math.round(Math.min(Math.max(value, 0), 1) * 100);
  return (
    <div className="flex items-center gap-2" title={`confidence ${pct}%`}>
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-surface-2 dark:bg-slate-800">
        <div className="h-full rounded-full bg-primary" style={{ width: `${pct}%` }} />
      </div>
      <span className="num text-2xs text-slate-500">{pct}%</span>
    </div>
  );
}

export function Stat({ label, value, tone, sub }:
  { label: string; value: ReactNode; tone?: 'long' | 'short' | 'warn' | 'success' | 'danger' | 'warning'; sub?: ReactNode }) {
  const color = tone === 'long' || tone === 'success' ? 'text-success'
    : tone === 'short' || tone === 'danger' ? 'text-danger'
    : tone === 'warn' || tone === 'warning' ? 'text-warning' : 'text-slate-900 dark:text-slate-100';
  return (
    <div className="card px-4 py-3">
      <div className="text-2xs font-semibold uppercase tracking-wider text-slate-500">{label}</div>
      <div className={`num mt-1 text-lg font-semibold ${color}`}>{value}</div>
      {sub && <div className="mt-0.5 text-2xs text-slate-500">{sub}</div>}
    </div>
  );
}

export function Section({ title, children, right }:
  { title: string; children: ReactNode; right?: ReactNode }) {
  return (
    <section className="card overflow-hidden">
      <header className="flex items-center justify-between border-b border-line bg-surface-2/50 px-4 py-2.5 dark:bg-slate-800/40">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</h2>
        {right}
      </header>
      {children}
    </section>
  );
}

export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="px-6 py-12 text-center">
      <div className="text-sm text-slate-500">{title}</div>
      {hint && <div className="mx-auto mt-1.5 max-w-md text-2xs text-slate-400">{hint}</div>}
    </div>
  );
}

export function Skeleton({ rows = 4 }: { rows?: number }) {
  return (
    <div className="space-y-2 p-4">
      {Array.from({ length: rows }, (_, i) => (
        <div key={i} className="h-6 animate-pulse rounded bg-surface-2 dark:bg-slate-800"
          style={{ opacity: 1 - i * 0.15 }} />
      ))}
    </div>
  );
}

/** Dependency-free sparkline for index-based series (e.g. R-curves). */
export function Sparkline({ data, width = 240, height = 48, stroke = 'var(--primary)', fill }:
  { data: number[]; width?: number; height?: number; stroke?: string; fill?: string }) {
  if (!data || data.length < 2) {
    return <div className="text-2xs text-slate-500">not enough data</div>;
  }
  const min = Math.min(...data);
  const max = Math.max(...data);
  const span = max - min || 1;
  const pts = data.map((v, i) =>
    `${((i / (data.length - 1)) * (width - 2) + 1).toFixed(1)},` +
    `${(height - 1 - ((v - min) / span) * (height - 2)).toFixed(1)}`).join(' ');
  const zeroY = height - 1 - ((0 - min) / span) * (height - 2);
  const fillD = `M1,${height - 1} L${pts.replace(/ /g, ' L')} L${width - 1},${height - 1} Z`;
  return (
    <svg width={width} height={height} className="block overflow-visible">
      {min < 0 && max > 0 && (
        <line x1="0" x2={width} y1={zeroY} y2={zeroY}
          stroke="rgba(148,163,184,0.25)" strokeDasharray="3 3" />
      )}
      {fill && <path d={fillD} fill={fill} opacity={0.25} />}
      <polyline points={pts} fill="none" stroke={stroke} strokeWidth="1.5" />
    </svg>
  );
}

export function Dot({ tone = 'neutral', pulse = false }:
  { tone?: 'neutral' | 'success' | 'danger' | 'warning' | 'primary'; pulse?: boolean }) {
  const map = {
    neutral: 'bg-slate-400',
    success: 'bg-success',
    danger: 'bg-danger',
    warning: 'bg-warning',
    primary: 'bg-primary',
  };
  return <span className={`chip-dot ${map[tone]} ${pulse ? 'animate-pulse' : ''}`} />;
}
