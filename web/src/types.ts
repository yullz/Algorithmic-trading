export interface TakeProfit {
  price: number;
  r: number;
  alloc: number;
}

export interface Plan {
  symbol: string;
  timeframe: string;
  side: 'LONG' | 'SHORT';
  entry: number;
  stop_loss: number;
  take_profits: TakeProfit[];
  leverage: number;
  qty: number;
  notional: number;
  margin: number;
  risk_amount: number;
  liquidation_price: number;
  reward_risk: number;
  expected_win_rate: number;
  expected_value_r: number;
  confidence: number;
  fees_estimate: number;
  warnings: string[];
  rationale: string[];
  explanation: string[];
  created_at: string;
  regime: string;
  families: string[];
  ml_prob: number | null;
  ml_weight: number;
  ml_contribs: string[];
  rank?: number;
  suppressed_by?: string;
}

export interface MarketTile {
  symbol: string;
  last: number;
  chg24h_pct: number;
  candidate: boolean;
  picked: boolean;
}

export interface Breadth {
  n: number;
  pct_above_ema50: number | null;
  pct_above_ema200: number | null;
  advancers: number;
  decliners: number;
  ad_ratio: number | null;
  risk_state: 'risk_on' | 'neutral' | 'risk_off';
}

export interface ScanResult {
  plans: Plan[];
  market: MarketTile[];
  suppressed_by_correlation: Plan[];
  universe_size: number;
  pairs_scanned: number;
  fetch_errors: number;
  candidates: number;
  btc_regime: string;
  breadth?: Breadth | null;
  scanned_at: string | null;
  duration_sec: number;
}

export interface OpenPosition {
  id: string;
  symbol: string;
  timeframe: string;
  side: 'LONG' | 'SHORT';
  entry: number;
  qty_initial: number;
  qty_open: number;
  stop: number;
  take_profits: [number, number, number, boolean][];
  leverage: number;
  margin: number;
  opened_at: string;
  plan: Plan;
  realized_pnl: number;
  unrealized_pnl: number;
  breakeven_moved: boolean;
  last_price: number;
}

export interface ClosedTrade {
  id: string;
  symbol: string;
  side: string;
  entry: number;
  exit_reason: string;
  pnl: number;
  r: number;
  win: boolean;
  opened_at: string;
  closed_at: string;
  timeframe: string;
}

export interface Positions {
  equity: number;
  mtm_equity: number;
  return_pct: number;
  open_positions: OpenPosition[];
  closed_trades: ClosedTrade[];
  equity_curve: [string, number][];
  consecutive_losses: number;
  day_anchor: { date: string; equity: number };
  updated_at: string;
}

export interface Config {
  exchange: string;
  universe_mode: string;
  universe_size: number;
  timeframes: string[];
  context_timeframe: string;
  equity: number;
  risk_per_trade_pct: number;
  max_leverage: number;
  default_leverage: number;
  max_concurrent_positions: number;
  max_daily_loss_pct: number;
  scan_interval_sec: number;
  calibrated: boolean;
  ml_enabled: boolean;
  execution_mode: string;
  testnet: boolean;
  live_enabled: boolean;
  live_attached: boolean;
}

export interface BacktestReport {
  summary: {
    trades?: number;
    win_rate?: number;
    expectancy_r?: number;
    profit_factor?: number;
    max_drawdown_r?: number;
    avg_win_r?: number;
    avg_loss_r?: number;
  };
  factor_win_rate?: Record<string, number>;
  kind_win_rate?: Record<string, number>;
  equity_curve: number[];
  n_trades?: number;
  symbols?: string[];
  timeframe?: string;
  source?: string;
}

export interface WalkforwardReport {
  in_sample?: Record<string, number>;
  out_of_sample?: Record<string, number>;
  folds?: Record<string, unknown>[];
  oos_equity_curve?: number[];
  verdict?: string;
  n_folds?: number;
}

export interface MLModelInfo {
  present: boolean;
  trusted?: boolean;
  n_train?: number;
  n_valid?: number;
  auc_valid?: number;
  brier_valid?: number;
  base_rate?: number;
  trained_at?: string;
  error?: string;
}

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export type LinePoint = { time: number; value: number };

export interface CandlesResponse {
  symbol: string;
  tf: string;
  candles: Candle[];
  sr_levels: { price: number; touches: number }[];
  overlays?: Record<string, LinePoint[]>;
  error?: string;
}

export interface ExposureBucket {
  name: string;
  long_notional: number;
  short_notional: number;
  net_notional: number;
  margin: number;
  count: number;
}

export interface ExposureSide {
  notional: number;
  margin: number;
  count: number;
}

export interface Exposure {
  sectors: ExposureBucket[];
  correlation_buckets: ExposureBucket[];
  sides: {
    long: ExposureSide;
    short: ExposureSide;
    net: number;
    gross: number;
  };
}

export interface WSMessage {
  type: 'hello' | 'scan' | 'positions' | 'exposure' | 'price_tick' | 'error';
  data: unknown;
  ts: string;
}

export interface HistorySignal {
  id: number;
  scan_id: number;
  timestamp: string | null;
  symbol: string;
  timeframe: string | null;
  side: 'LONG' | 'SHORT' | string;
  kind: string;
  entry: number | null;
  stop: number | null;
  confidence: number | null;
  score: number | null;
  win_rate: number | null;
  expected_value_r: number | null;
  rationale_json: string | null;
  btc_regime: string | null;
}

export interface HistoryTrade {
  id: number;
  scan_id: number | null;
  signal_id: number | null;
  timestamp: string | null;
  symbol: string;
  side: 'LONG' | 'SHORT' | string;
  entry: number | null;
  stop: number | null;
  qty: number | null;
  leverage: number | null;
  margin: number | null;
  status: string | null;
  mtm_pnl: number | null;
  outcome: string | null;
  realized_r: number | null;
  fees_r: number | null;
  closed_at: string | null;
}

export interface HistoryBreakdown {
  month?: string;
  btc_regime?: string;
  kind?: string;
  trades: number;
  avg_r: number | null;
  wins: number;
}

export interface HistorySummary {
  total_scans: number;
  total_signals: number;
  total_trades: number;
  closed_trades: number;
  win_rate: number | null;
  avg_realized_r: number | null;
  by_month: HistoryBreakdown[];
  by_regime: HistoryBreakdown[];
  by_setup_kind: HistoryBreakdown[];
}

export interface HealthCircuitBreakers {
  kill_switch: boolean;
  daily_loss_pct: number;
  daily_loss_triggered: boolean;
  consecutive_losses: number;
  losing_streak_triggered: boolean;
}

export interface Health {
  last_scan_age_sec: number | null;
  data_fresh: boolean;
  calibration_stale: boolean;
  model_trusted: boolean;
  circuit_breakers: HealthCircuitBreakers;
  paused: boolean;
}

export interface ErrorEntry {
  id: string;
  source: 'ws' | 'rest';
  message: string;
  ts: string;
}
