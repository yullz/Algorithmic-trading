import { useEffect, useRef, useState } from 'react';
import type {
  Config, ErrorEntry, Exposure, Health, HistorySignal, HistorySummary, HistoryTrade,
  Positions, ScanResult, WSMessage,
} from './types';

const MAX_ERRORS = 50;
let errorLog: ErrorEntry[] = [];
const errorSubs = new Set<(log: ErrorEntry[]) => void>();

export function pushError(source: ErrorEntry['source'], message: string) {
  const entry: ErrorEntry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
    source,
    message,
    ts: new Date().toISOString(),
  };
  errorLog = [entry, ...errorLog].slice(0, MAX_ERRORS);
  errorSubs.forEach(fn => fn(errorLog));
}

export function useErrors(): ErrorEntry[] {
  const [log, setLog] = useState<ErrorEntry[]>(errorLog);
  useEffect(() => {
    errorSubs.add(setLog);
    return () => { errorSubs.delete(setLog); };
  }, []);
  return log;
}

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`${path}: ${r.status} ${text}`);
  }
  return r.json() as Promise<T>;
}

export async function post<T>(path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const text = await r.text().catch(() => '');
    throw new Error(`${path}: ${r.status} ${text}`);
  }
  return r.json() as Promise<T>;
}

export function useConfig(): Config | null {
  const [cfg, setCfg] = useState<Config | null>(null);
  useEffect(() => {
    get<Config>('/api/config')
      .then(setCfg)
      .catch(e => { pushError('rest', `config: ${e.message}`); setCfg(null); });
  }, []);
  return cfg;
}

export interface LiveState {
  scan: ScanResult | null;
  positions: Positions | null;
  exposure: Exposure | null;
  connected: boolean;
  paused: boolean;
  pause: () => Promise<void>;
  resume: () => Promise<void>;
  closePosition: (symbol: string) => Promise<void>;
}

/** Apply a live price tick to open positions: refresh last_price, unrealized
 *  PnL, and the mark-to-market equity so the UI moves between scans. */
function applyPriceTick(prev: Positions | null, symbol: string,
                        price: number): Positions | null {
  if (!prev) return prev;
  let changed = false;
  const open_positions = prev.open_positions.map(p => {
    if (p.symbol !== symbol) return p;
    changed = true;
    const sign = p.side === 'LONG' ? 1 : -1;
    return { ...p, last_price: price,
             unrealized_pnl: sign * (price - p.entry) * p.qty_open };
  });
  if (!changed) return prev;
  const mtm_equity = prev.equity
    + open_positions.reduce((s, p) => s + p.unrealized_pnl, 0);
  return { ...prev, open_positions, mtm_equity };
}

/** WebSocket feed with auto-reconnect; falls back to polling while down. */
export function useLive(): LiveState {
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [positions, setPositions] = useState<Positions | null>(null);
  const [exposure, setExposure] = useState<Exposure | null>(null);
  const [connected, setConnected] = useState(false);
  const [paused, setPaused] = useState(false);
  const backoff = useRef(1000);

  // Sync initial pause state with the server.
  useEffect(() => {
    get<Health>('/api/health')
      .then(h => setPaused(h.paused))
      .catch(() => {});
  }, []);

  const pause = async () => {
    try {
      await post<{ paused: boolean }>('/api/control/pause');
      setPaused(true);
    } catch (e) {
      pushError('rest', `pause failed: ${(e as Error).message}`);
      throw e;
    }
  };

  const resume = async () => {
    try {
      await post<{ paused: boolean }>('/api/control/resume');
      setPaused(false);
    } catch (e) {
      pushError('rest', `resume failed: ${(e as Error).message}`);
      throw e;
    }
  };

  const closePosition = async (symbol: string) => {
    try {
      await post<{ status: string; symbol: string; price: number }>(`/api/positions/${encodeURIComponent(symbol)}/close`);
    } catch (e) {
      pushError('rest', `close ${symbol}: ${(e as Error).message}`);
      throw e;
    }
  };

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let ping: number | undefined;
    let reconnect: number | undefined;
    let poll: number | undefined;

    const refresh = () => {
      get<ScanResult>('/api/signals').then(s => { if (s?.plans) setScan(s); }).catch(() => {});
      get<Positions>('/api/positions').then(setPositions).catch(() => {});
      get<Exposure>('/api/exposure').then(setExposure).catch(() => {});
    };
    refresh();

    const connect = () => {
      if (closed) return;
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.onopen = () => {
        setConnected(true);
        backoff.current = 1000;
        if (poll) { clearInterval(poll); poll = undefined; }
        ping = window.setInterval(() => ws?.readyState === 1 && ws.send('ping'), 20000);
      };
      ws.onmessage = ev => {
        try {
          const msg = JSON.parse(ev.data) as WSMessage;
          if (msg.type === 'hello') {
            const d = msg.data as { signals?: ScanResult; positions?: Positions; exposure?: Exposure };
            if (d.signals?.plans) setScan(d.signals);
            if (d.positions) setPositions(d.positions);
            if (d.exposure) setExposure(d.exposure);
          } else if (msg.type === 'scan') {
            setScan(msg.data as ScanResult);
          } else if (msg.type === 'positions') {
            setPositions(msg.data as Positions);
          } else if (msg.type === 'exposure') {
            setExposure(msg.data as Exposure);
          } else if (msg.type === 'price_tick') {
            const d = msg.data as { symbol: string; price: number };
            setPositions(prev => applyPriceTick(prev, d.symbol, d.price));
            setScan(prev => (prev?.market
              ? { ...prev, market: prev.market.map(m =>
                  m.symbol === d.symbol ? { ...m, last: d.price } : m) }
              : prev));
          } else if (msg.type === 'error') {
            const d = msg.data as { message?: string };
            pushError('ws', d.message || 'Unknown server error');
          }
        } catch { /* malformed frame — ignore */ }
      };
      ws.onclose = () => {
        setConnected(false);
        if (ping) clearInterval(ping);
        if (closed) return;
        if (!poll) poll = window.setInterval(refresh, 30000);
        reconnect = window.setTimeout(connect, backoff.current);
        backoff.current = Math.min(backoff.current * 2, 30000);
      };
      ws.onerror = () => ws?.close();
    };
    connect();

    return () => {
      closed = true;
      if (ping) clearInterval(ping);
      if (reconnect) clearTimeout(reconnect);
      if (poll) clearInterval(poll);
      ws?.close();
    };
  }, []);

  return { scan, positions, exposure, connected, paused, pause, resume, closePosition };
}

export interface HistorySignalFilters {
  symbol?: string;
  timeframe?: string;
  side?: string;
  regime?: string;
  from?: string;
  to?: string;
  limit?: number;
}

export interface HistoryTradeFilters {
  symbol?: string;
  outcome?: string;
  from?: string;
  to?: string;
  limit?: number;
}

function filtersKey(f: HistorySignalFilters | HistoryTradeFilters): string {
  return Object.entries(f)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}=${v}`)
    .join('&');
}

function buildQs(f: HistorySignalFilters | HistoryTradeFilters): string {
  const qs = new URLSearchParams();
  Object.entries(f).forEach(([k, v]) => {
    if (v === undefined || v === '' || v === null) return;
    qs.append(k, String(v));
  });
  const s = qs.toString();
  return s ? `?${s}` : '';
}

export function useHistorySignals(filters: HistorySignalFilters = {}): HistorySignal[] | null {
  const [data, setData] = useState<HistorySignal[] | null>(null);
  const key = filtersKey(filters);

  useEffect(() => {
    const path = `/api/history/signals${buildQs(filters)}`;
    const load = () => get<{ signals: HistorySignal[] }>(path)
      .then(r => setData(r.signals))
      .catch(e => { pushError('rest', `history signals: ${e.message}`); });
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, [key]);

  return data;
}

export function useHistoryTrades(filters: HistoryTradeFilters = {}): HistoryTrade[] | null {
  const [data, setData] = useState<HistoryTrade[] | null>(null);
  const key = filtersKey(filters);

  useEffect(() => {
    const path = `/api/history/trades${buildQs(filters)}`;
    const load = () => get<{ trades: HistoryTrade[] }>(path)
      .then(r => setData(r.trades))
      .catch(e => { pushError('rest', `history trades: ${e.message}`); });
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, [key]);

  return data;
}

export function useHistorySummary(): HistorySummary | null {
  const [data, setData] = useState<HistorySummary | null>(null);

  useEffect(() => {
    const load = () => get<HistorySummary>('/api/history/summary')
      .then(setData)
      .catch(e => { pushError('rest', `history summary: ${e.message}`); });
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  return data;
}

export function useHealth(): Health | null {
  const [data, setData] = useState<Health | null>(null);

  useEffect(() => {
    const load = () => get<Health>('/api/health')
      .then(setData)
      .catch(e => { pushError('rest', `health: ${e.message}`); });
    load();
    const id = setInterval(load, 10000);
    return () => clearInterval(id);
  }, []);

  return data;
}

/** Re-render every `ms` so "Xs ago" chips stay current. */
export function useTick(ms = 5000): number {
  const [t, setT] = useState(Date.now());
  useEffect(() => {
    const id = setInterval(() => setT(Date.now()), ms);
    return () => clearInterval(id);
  }, [ms]);
  return t;
}
