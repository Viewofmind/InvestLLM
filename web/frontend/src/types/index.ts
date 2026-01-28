// Trading Types
export type TradingMode = 'paper' | 'live';
export type OrderSide = 'BUY' | 'SELL';
export type OrderStatus = 'pending' | 'filled' | 'rejected' | 'cancelled';
export type SignalType = 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL';

export interface Position {
  symbol: string;
  quantity: number;
  avg_price: number;
  current_price: number;
  pnl: number;
  pnl_pct: number;
  side: OrderSide;
  entry_time: string;
}

export interface Order {
  id: string;
  symbol: string;
  side: OrderSide;
  quantity: number;
  price: number;
  status: OrderStatus;
  timestamp: string;
  filled_price?: number;
  message?: string;
}

export interface Signal {
  id: string;
  symbol: string;
  signal: SignalType;
  strength: number;
  confidence: number;
  source: string;
  timestamp: string;
  executed: boolean;
  price_target?: number;
  stop_loss?: number;
}

export interface RiskMetrics {
  current_drawdown: number;
  max_drawdown: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  position_count: number;
  total_exposure: number;
  exposure_pct: number;
  circuit_breaker_active: boolean;
}

export interface RiskAlert {
  id: string;
  level: 'warning' | 'critical';
  message: string;
  timestamp: string;
  acknowledged: boolean;
}

export interface PortfolioSummary {
  total_value: number;
  cash: number;
  positions_value: number;
  total_pnl: number;
  total_pnl_pct: number;
  day_pnl: number;
  day_pnl_pct: number;
}

export interface PerformanceData {
  timestamps: string[];
  values: number[];
  benchmark?: number[];
}

export interface Settings {
  mode: TradingMode;
  initial_capital: number;
  is_running: boolean;
  risk_limits: {
    max_position_pct: number;
    max_sector_pct: number;
    max_drawdown_pct: number;
    daily_loss_limit_pct: number;
  };
  api_configured: {
    zerodha: boolean;
    firecrawl: boolean;
    gemini: boolean;
  };
}

// WebSocket Message Types
export interface WSMessage {
  type: 'portfolio' | 'positions' | 'signal' | 'risk' | 'alert' | 'trade' | 'error';
  data: unknown;
  timestamp: string;
}
