import { useEffect, useState } from 'react';
import AnalyticsView from './AnalyticsView';
import ErrorLog from './ErrorLog';
import HistoryView from './HistoryView';
import PortfolioView from './PortfolioView';
import ScannerView from './ScannerView';
import SignalDetail from './SignalDetail';
import StatusBar from './StatusBar';
import { useConfig, useErrors, useHealth, useLive } from './api';
import type { Plan } from './types';

type Tab = 'scanner' | 'portfolio' | 'analytics' | 'history';

interface Selection {
  symbol: string;
  tf: string;
  plan?: Plan;
}

const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: 'scanner', label: 'Scanner', icon: '◎' },
  { id: 'portfolio', label: 'Portfolio', icon: '◧' },
  { id: 'analytics', label: 'Analytics', icon: '◈' },
  { id: 'history', label: 'History', icon: '◼' },
];

function useTheme() {
  const [dark, setDark] = useState(() => {
    const stored = typeof localStorage !== 'undefined' ? localStorage.getItem('edge:theme') : null;
    return stored ? stored === 'dark' : true;
  });

  useEffect(() => {
    if (dark) {
      document.documentElement.classList.add('dark');
      document.documentElement.style.colorScheme = 'dark';
    } else {
      document.documentElement.classList.remove('dark');
      document.documentElement.style.colorScheme = 'light';
    }
    localStorage.setItem('edge:theme', dark ? 'dark' : 'light');
  }, [dark]);

  return { dark, setDark, toggle: () => setDark(d => !d) } as const;
}

export default function App() {
  const cfg = useConfig();
  const { scan, positions, exposure, connected, paused, pause, resume, closePosition } = useLive();
  const health = useHealth();
  const errors = useErrors();
  const [tab, setTab] = useState<Tab>('scanner');
  const [sel, setSel] = useState<Selection | null>(null);
  const [errorLogOpen, setErrorLogOpen] = useState(false);
  const { dark, toggle } = useTheme();

  return (
    <div className="flex min-h-screen flex-col bg-surface text-neutral-700 dark:bg-neutral-950 dark:text-neutral-300">
      <StatusBar
        cfg={cfg}
        scan={scan}
        positions={positions}
        connected={connected}
        health={health}
        errorCount={errors.length}
        onToggleErrors={() => setErrorLogOpen(o => !o)}
      />

      {/* Top navigation */}
      <nav className="border-b border-line bg-surface-1/80 backdrop-blur dark:bg-neutral-900/80">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-2">
          <div className="flex items-center gap-1">
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className={`btn-tab ${tab === t.id ? 'btn-tab-active ring-1 ring-line-strong' : ''}`}
                aria-current={tab === t.id ? 'page' : undefined}
              >
                <span className="text-primary">{t.icon}</span>
                <span className="capitalize">{t.label}</span>
              </button>
            ))}
          </div>

          <button
            onClick={toggle}
            className="btn-icon"
            title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {dark ? '☀' : '☾'}
          </button>
        </div>
      </nav>

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-5">
        {tab === 'scanner' && (
          <ScannerView
            scan={scan}
            paused={paused}
            onPause={pause}
            onResume={resume}
            onSelect={(symbol, tf, plan) => setSel({ symbol, tf, plan })}
          />
        )}
        {tab === 'portfolio' && <PortfolioView positions={positions} exposure={exposure} onClose={closePosition} />}
        {tab === 'analytics' && <AnalyticsView />}
        {tab === 'history' && <HistoryView />}
      </main>

      {sel && (
        <SignalDetail symbol={sel.symbol} tf={sel.tf} plan={sel.plan} onClose={() => setSel(null)} />
      )}

      <ErrorLog open={errorLogOpen} onClose={() => setErrorLogOpen(false)} />

      <footer className="border-t border-line bg-surface-1 px-4 py-3 text-center text-2xs text-neutral-500 dark:bg-neutral-900/60">
        Signals are calibrated estimates, not predictions. Leverage can liquidate you.
        Nothing here is financial advice.
      </footer>
    </div>
  );
}
