import { fmtMoney, fmtPct, fmtSigned } from './format';
import { Section } from './ui';
import type { Exposure } from './types';

export default function ExposureHeatmap({ exposure }:
  { exposure: Exposure | null }) {
  if (!exposure) {
    return (
      <Section title="Exposure">
        <div className="px-6 py-8 text-center text-sm text-neutral-500">
          Waiting for portfolio exposure snapshot…
        </div>
      </Section>
    );
  }

  return (
    <Section title="Exposure heatmap">
      <div className="grid gap-4 p-4 lg:grid-cols-3">
        <SectorCard sectors={exposure.sectors} />
        <CorrelationCard buckets={exposure.correlation_buckets} />
        <SideCard sides={exposure.sides} />
      </div>
    </Section>
  );
}

function SectorCard({ sectors }: { sectors: Exposure['sectors'] }) {
  const totalNet = sectors.reduce((s, b) => s + Math.abs(b.net_notional), 0) || 1;
  return (
    <div className="card p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-500">
        Sector
      </div>
      {sectors.length === 0 ? (
        <div className="text-sm text-neutral-500">No open positions</div>
      ) : (
        <div className="space-y-3">
          {sectors.map(s => (
            <div key={s.name}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="font-medium text-neutral-900 dark:text-neutral-100">{s.name}</span>
                <span className="num text-2xs text-neutral-500">{s.count} pos</span>
              </div>
              <div className="relative h-2 w-full overflow-hidden rounded-full bg-surface-2 dark:bg-neutral-800">
                <div
                  className="absolute top-0 h-full bg-success"
                  style={{
                    left: `${Math.max(0, Math.min(50, (s.long_notional / totalNet) * 50))}%`,
                    width: `${Math.max(0, Math.min(50, (s.long_notional / totalNet) * 50))}%`,
                  }}
                />
                <div
                  className="absolute top-0 h-full bg-danger"
                  style={{
                    right: `${Math.max(0, Math.min(50, (s.short_notional / totalNet) * 50))}%`,
                    width: `${Math.max(0, Math.min(50, (s.short_notional / totalNet) * 50))}%`,
                  }}
                />
              </div>
              <div className="mt-1 flex justify-between text-2xs">
                <span className="text-success">{fmtMoney(s.long_notional)}</span>
                <span className="text-danger">{fmtMoney(s.short_notional)}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function CorrelationCard({ buckets }: { buckets: Exposure['correlation_buckets'] }) {
  const total = buckets.reduce((s, b) => s + Math.abs(b.net_notional), 0) || 1;
  const colors: Record<string, string> = {
    low: 'bg-success',
    medium: 'bg-warning',
    high: 'bg-danger',
  };
  const labels: Record<string, string> = {
    low: 'Low (<0.4)',
    medium: 'Medium (0.4–0.7)',
    high: 'High (>0.7)',
  };
  return (
    <div className="card p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-500">
        BTC correlation
      </div>
      <div className="space-y-3">
        {buckets.map(b => {
          const pct = Math.abs(b.net_notional) / total;
          return (
            <div key={b.name}>
              <div className="mb-1 flex items-center justify-between text-sm">
                <span className="font-medium text-neutral-900 dark:text-neutral-100">{labels[b.name] || b.name}</span>
                <span className="num text-2xs text-neutral-500">{b.count} pos</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2 dark:bg-neutral-800">
                <div
                  className={`h-full rounded-full ${colors[b.name] || 'bg-primary'}`}
                  style={{ width: `${Math.min(100, pct * 100)}%` }}
                />
              </div>
              <div className="mt-1 flex justify-between text-2xs">
                <span className={b.net_notional >= 0 ? 'text-success' : 'text-danger'}>
                  {fmtSigned(b.net_notional)}
                </span>
                <span className="text-neutral-500">{fmtPct(pct)}</span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function SideCard({ sides }: { sides: Exposure['sides'] }) {
  const { long, short, net, gross } = sides;
  const longShare = gross > 0 ? long.notional / gross : 0;
  const shortShare = gross > 0 ? short.notional / gross : 0;
  return (
    <div className="card p-4">
      <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-neutral-500">
        Side summary
      </div>
      <div className="space-y-4">
        <div>
          <div className="mb-1 flex justify-between text-sm">
            <span className="font-medium text-success">Long</span>
            <span className="num text-neutral-900 dark:text-neutral-100">{fmtMoney(long.notional)}</span>
          </div>
          <div className="mb-1 flex justify-between text-sm">
            <span className="font-medium text-danger">Short</span>
            <span className="num text-neutral-900 dark:text-neutral-100">{fmtMoney(short.notional)}</span>
          </div>
          <div className="h-2.5 w-full overflow-hidden rounded-full bg-surface-2 dark:bg-neutral-800">
            <div className="flex h-full">
              <div className="h-full bg-success" style={{ width: `${longShare * 100}%` }} />
              <div className="h-full bg-danger" style={{ width: `${shortShare * 100}%` }} />
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="rounded-md bg-surface-2 px-2 py-1.5 dark:bg-neutral-800/50">
            <div className="text-2xs text-neutral-500">Net exposure</div>
            <div className={`num font-semibold ${net >= 0 ? 'text-success' : 'text-danger'}`}>
              {fmtSigned(net)}
            </div>
          </div>
          <div className="rounded-md bg-surface-2 px-2 py-1.5 dark:bg-neutral-800/50">
            <div className="text-2xs text-neutral-500">Gross exposure</div>
            <div className="num font-semibold text-neutral-900 dark:text-neutral-100">
              {fmtMoney(gross)}
            </div>
          </div>
          <div className="rounded-md bg-surface-2 px-2 py-1.5 dark:bg-neutral-800/50">
            <div className="text-2xs text-neutral-500">Long margin</div>
            <div className="num font-semibold text-success">{fmtMoney(long.margin)}</div>
          </div>
          <div className="rounded-md bg-surface-2 px-2 py-1.5 dark:bg-neutral-800/50">
            <div className="text-2xs text-neutral-500">Short margin</div>
            <div className="num font-semibold text-danger">{fmtMoney(short.margin)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
