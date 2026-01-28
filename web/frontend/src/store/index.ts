import { create } from 'zustand';
import type {
  Position,
  Signal,
  RiskMetrics,
  RiskAlert,
  PortfolioSummary,
  Settings,
  Order,
} from '../types';
import { portfolioAPI, signalsAPI, riskAPI, settingsAPI } from '../services/api';

interface TradingStore {
  // State
  portfolio: PortfolioSummary | null;
  positions: Position[];
  signals: Signal[];
  risk: RiskMetrics | null;
  alerts: RiskAlert[];
  settings: Settings | null;
  recentOrders: Order[];
  isLoading: boolean;
  error: string | null;

  // Actions
  fetchPortfolio: () => Promise<void>;
  fetchPositions: () => Promise<void>;
  fetchSignals: () => Promise<void>;
  fetchRisk: () => Promise<void>;
  fetchAlerts: () => Promise<void>;
  fetchSettings: () => Promise<void>;
  fetchAll: () => Promise<void>;

  // Update from WebSocket
  updatePortfolio: (data: PortfolioSummary) => void;
  updatePositions: (data: Position[]) => void;
  addSignal: (signal: Signal) => void;
  updateRisk: (data: RiskMetrics) => void;
  addAlert: (alert: RiskAlert) => void;
  addOrder: (order: Order) => void;

  // Utilities
  setError: (error: string | null) => void;
  clearError: () => void;
}

export const useTradingStore = create<TradingStore>((set, get) => ({
  // Initial state
  portfolio: null,
  positions: [],
  signals: [],
  risk: null,
  alerts: [],
  settings: null,
  recentOrders: [],
  isLoading: false,
  error: null,

  // Fetch actions
  fetchPortfolio: async () => {
    try {
      const data = await portfolioAPI.getSummary();
      set({ portfolio: data });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  fetchPositions: async () => {
    try {
      const data = await portfolioAPI.getPositions();
      set({ positions: data });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  fetchSignals: async () => {
    try {
      const data = await signalsAPI.getSignals(true);
      set({ signals: data });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  fetchRisk: async () => {
    try {
      const data = await riskAPI.getSummary();
      set({ risk: data });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  fetchAlerts: async () => {
    try {
      const data = await riskAPI.getAlerts();
      set({ alerts: data });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  fetchSettings: async () => {
    try {
      const data = await settingsAPI.getSettings();
      set({ settings: data });
    } catch (error) {
      set({ error: (error as Error).message });
    }
  },

  fetchAll: async () => {
    set({ isLoading: true, error: null });
    await Promise.all([
      get().fetchPortfolio(),
      get().fetchPositions(),
      get().fetchSignals(),
      get().fetchRisk(),
      get().fetchAlerts(),
      get().fetchSettings(),
    ]);
    set({ isLoading: false });
  },

  // WebSocket updates
  updatePortfolio: (data) => set({ portfolio: data }),

  updatePositions: (data) => set({ positions: data }),

  addSignal: (signal) =>
    set((state) => ({
      signals: [signal, ...state.signals].slice(0, 50),
    })),

  updateRisk: (data) => set({ risk: data }),

  addAlert: (alert) =>
    set((state) => ({
      alerts: [alert, ...state.alerts].slice(0, 20),
    })),

  addOrder: (order) =>
    set((state) => ({
      recentOrders: [order, ...state.recentOrders].slice(0, 20),
    })),

  // Utilities
  setError: (error) => set({ error }),
  clearError: () => set({ error: null }),
}));
