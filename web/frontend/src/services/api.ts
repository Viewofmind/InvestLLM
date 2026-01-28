import type {
  Position,
  Order,
  Signal,
  RiskMetrics,
  RiskAlert,
  PortfolioSummary,
  PerformanceData,
  Settings,
  OrderSide,
} from '../types';

const API_BASE = '/api';

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Portfolio API
export const portfolioAPI = {
  getSummary: () => fetchJSON<PortfolioSummary>('/portfolio/'),

  getPositions: () => fetchJSON<Position[]>('/portfolio/positions'),

  getPerformance: (days = 30) =>
    fetchJSON<PerformanceData>(`/portfolio/performance?days=${days}`),

  getHistory: (limit = 50) =>
    fetchJSON<Order[]>(`/portfolio/history?limit=${limit}`),
};

// Trading API
export const tradingAPI = {
  placeOrder: (symbol: string, side: OrderSide, quantity: number, price?: number) =>
    fetchJSON<Order>('/trading/order', {
      method: 'POST',
      body: JSON.stringify({ symbol, side, quantity, price }),
    }),

  closePosition: (symbol: string) =>
    fetchJSON<{ success: boolean; order: Order }>(`/trading/close/${symbol}`, {
      method: 'POST',
    }),

  closeAllPositions: () =>
    fetchJSON<{ success: boolean; closed: number }>('/trading/close-all', {
      method: 'POST',
    }),

  startTrading: () =>
    fetchJSON<{ success: boolean; message: string }>('/trading/start', {
      method: 'POST',
    }),

  stopTrading: () =>
    fetchJSON<{ success: boolean; message: string }>('/trading/stop', {
      method: 'POST',
    }),
};

// Signals API
export const signalsAPI = {
  getSignals: (pending_only = false) =>
    fetchJSON<Signal[]>(`/signals/?pending_only=${pending_only}`),

  executeSignal: (signalId: string) =>
    fetchJSON<{ success: boolean; order: Order }>(`/signals/${signalId}/execute`, {
      method: 'POST',
    }),

  dismissSignal: (signalId: string) =>
    fetchJSON<{ success: boolean }>(`/signals/${signalId}/dismiss`, {
      method: 'POST',
    }),
};

// Risk API
export const riskAPI = {
  getSummary: () => fetchJSON<RiskMetrics>('/risk/'),

  getAlerts: () => fetchJSON<RiskAlert[]>('/risk/alerts'),

  acknowledgeAlert: (alertId: string) =>
    fetchJSON<{ success: boolean }>(`/risk/alerts/${alertId}/acknowledge`, {
      method: 'POST',
    }),

  getLimits: () => fetchJSON<Settings['risk_limits']>('/risk/limits'),
};

// Settings API
export const settingsAPI = {
  getSettings: () => fetchJSON<Settings>('/settings/'),

  setMode: (mode: 'paper' | 'live') =>
    fetchJSON<{ success: boolean; mode: string }>('/settings/mode', {
      method: 'POST',
      body: JSON.stringify({ mode }),
    }),

  setCapital: (capital: number) =>
    fetchJSON<{ success: boolean; capital: number }>('/settings/capital', {
      method: 'POST',
      body: JSON.stringify({ capital }),
    }),

  reset: () =>
    fetchJSON<{ success: boolean; message: string }>('/settings/reset', {
      method: 'POST',
    }),
};
