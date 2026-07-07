export function fmtPrice(p: number | null | undefined): string {
  if (p === null || p === undefined || !isFinite(p)) return '—';
  if (p >= 100) return p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (p >= 1) return p.toFixed(4);
  return p.toPrecision(6);
}

export function fmtMoney(v: number | null | undefined): string {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return v.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

export function fmtPct(v: number | null | undefined, dp = 1): string {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return `${(v * 100).toFixed(dp)}%`;
}

export function fmtR(v: number | null | undefined): string {
  if (v === null || v === undefined || !isFinite(v)) return '—';
  return `${v >= 0 ? '+' : ''}${v.toFixed(2)}R`;
}

export function fmtSigned(v: number, dp = 2): string {
  return `${v >= 0 ? '+' : ''}${v.toFixed(dp)}`;
}

export function timeAgo(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return 'never';
  const t = new Date(iso).getTime();
  if (isNaN(t)) return '—';
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 90) return `${s}s ago`;
  if (s < 5400) return `${Math.round(s / 60)}m ago`;
  if (s < 172800) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function secondsSince(iso: string | null | undefined, now = Date.now()): number {
  if (!iso) return Infinity;
  const t = new Date(iso).getTime();
  return isNaN(t) ? Infinity : (now - t) / 1000;
}

/** 24h % change -> heat color (green↔red through neutral slate). */
export function heatColor(pct: number): string {
  const a = Math.min(Math.abs(pct) / 10, 1);
  return pct >= 0
    ? `rgba(16,185,129,${0.08 + 0.45 * a})`
    : `rgba(244,63,94,${0.08 + 0.45 * a})`;
}
