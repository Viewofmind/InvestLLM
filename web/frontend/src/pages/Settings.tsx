import { useState } from 'react';
import { useTradingStore } from '../store';
import { tradingAPI, settingsAPI } from '../services/api';
import {
  Settings as SettingsIcon,
  Play,
  Square,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  Loader2,
} from 'lucide-react';
import clsx from 'clsx';
import { formatCurrency } from '../utils/format';

export default function Settings() {
  const { settings, fetchSettings, fetchPortfolio, fetchPositions } = useTradingStore();

  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [isSwitchingMode, setIsSwitchingMode] = useState(false);
  const [capitalInput, setCapitalInput] = useState('');
  const [isSettingCapital, setIsSettingCapital] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleStartTrading = async () => {
    setIsStarting(true);
    setMessage(null);
    try {
      await tradingAPI.startTrading();
      await fetchSettings();
      setMessage({ type: 'success', text: 'Trading started successfully' });
    } catch (error) {
      setMessage({ type: 'error', text: (error as Error).message });
    } finally {
      setIsStarting(false);
    }
  };

  const handleStopTrading = async () => {
    setIsStopping(true);
    setMessage(null);
    try {
      await tradingAPI.stopTrading();
      await fetchSettings();
      setMessage({ type: 'success', text: 'Trading stopped' });
    } catch (error) {
      setMessage({ type: 'error', text: (error as Error).message });
    } finally {
      setIsStopping(false);
    }
  };

  const handleSwitchMode = async (mode: 'paper' | 'live') => {
    if (mode === 'live') {
      const confirmed = window.confirm(
        'WARNING: You are about to switch to LIVE trading mode. Real money will be used. Are you sure?'
      );
      if (!confirmed) return;
    }

    setIsSwitchingMode(true);
    setMessage(null);
    try {
      await settingsAPI.setMode(mode);
      await fetchSettings();
      setMessage({ type: 'success', text: `Switched to ${mode} trading mode` });
    } catch (error) {
      setMessage({ type: 'error', text: (error as Error).message });
    } finally {
      setIsSwitchingMode(false);
    }
  };

  const handleReset = async () => {
    const confirmed = window.confirm(
      'This will reset your paper trading account. All positions will be closed and P&L will be reset. Continue?'
    );
    if (!confirmed) return;

    setIsResetting(true);
    setMessage(null);
    try {
      await settingsAPI.reset();
      await Promise.all([fetchSettings(), fetchPortfolio(), fetchPositions()]);
      setMessage({ type: 'success', text: 'Paper trading account reset successfully' });
    } catch (error) {
      setMessage({ type: 'error', text: (error as Error).message });
    } finally {
      setIsResetting(false);
    }
  };

  const handleSetCapital = async () => {
    const capital = parseFloat(capitalInput);
    if (isNaN(capital) || capital <= 0) {
      setMessage({ type: 'error', text: 'Please enter a valid capital amount' });
      return;
    }

    setIsSettingCapital(true);
    setMessage(null);
    try {
      await settingsAPI.setCapital(capital);
      await Promise.all([fetchSettings(), fetchPortfolio()]);
      setCapitalInput('');
      setMessage({ type: 'success', text: `Capital set to ${formatCurrency(capital)}` });
    } catch (error) {
      setMessage({ type: 'error', text: (error as Error).message });
    } finally {
      setIsSettingCapital(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Settings</h1>
        <p className="text-slate-500 mt-1">Configure trading mode and parameters</p>
      </div>

      {/* Message Alert */}
      {message && (
        <div
          className={clsx(
            'flex items-center gap-3 px-4 py-3 rounded-lg',
            message.type === 'success'
              ? 'bg-emerald-50 border border-emerald-200 text-emerald-700'
              : 'bg-red-50 border border-red-200 text-red-700'
          )}
        >
          {message.type === 'success' ? (
            <CheckCircle className="h-5 w-5" />
          ) : (
            <AlertTriangle className="h-5 w-5" />
          )}
          {message.text}
        </div>
      )}

      {/* Trading Mode */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <div className="flex items-center gap-3 mb-6">
          <SettingsIcon className="h-6 w-6 text-slate-400" />
          <h2 className="text-lg font-semibold text-slate-800">Trading Mode</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <button
            onClick={() => handleSwitchMode('paper')}
            disabled={isSwitchingMode || settings?.mode === 'paper'}
            className={clsx(
              'p-6 rounded-xl border-2 text-left transition-all',
              settings?.mode === 'paper'
                ? 'border-emerald-500 bg-emerald-50'
                : 'border-slate-200 hover:border-slate-300'
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-lg text-slate-800">Paper Trading</h3>
              {settings?.mode === 'paper' && (
                <span className="px-2 py-1 bg-emerald-100 text-emerald-700 text-xs font-medium rounded">
                  Active
                </span>
              )}
            </div>
            <p className="text-sm text-slate-500">
              Practice trading with virtual money. No real transactions.
            </p>
          </button>

          <button
            onClick={() => handleSwitchMode('live')}
            disabled={isSwitchingMode || settings?.mode === 'live'}
            className={clsx(
              'p-6 rounded-xl border-2 text-left transition-all',
              settings?.mode === 'live'
                ? 'border-red-500 bg-red-50'
                : 'border-slate-200 hover:border-slate-300'
            )}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="font-semibold text-lg text-slate-800">Live Trading</h3>
              {settings?.mode === 'live' && (
                <span className="px-2 py-1 bg-red-100 text-red-700 text-xs font-medium rounded">
                  Active
                </span>
              )}
            </div>
            <p className="text-sm text-slate-500">
              Real trading with actual money via Zerodha Kite.
            </p>
          </button>
        </div>

        {/* Trading Controls */}
        <div className="flex items-center gap-4 pt-4 border-t border-slate-100">
          <p className="text-sm text-slate-500">
            Trading Status:{' '}
            <span
              className={clsx(
                'font-medium',
                settings?.is_running ? 'text-emerald-600' : 'text-slate-600'
              )}
            >
              {settings?.is_running ? 'Running' : 'Stopped'}
            </span>
          </p>

          {settings?.is_running ? (
            <button
              onClick={handleStopTrading}
              disabled={isStopping}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50"
            >
              {isStopping ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Square className="h-4 w-4" />
              )}
              Stop Trading
            </button>
          ) : (
            <button
              onClick={handleStartTrading}
              disabled={isStarting}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:opacity-50"
            >
              {isStarting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Play className="h-4 w-4" />
              )}
              Start Trading
            </button>
          )}
        </div>
      </div>

      {/* Paper Trading Capital (only show in paper mode) */}
      {settings?.mode === 'paper' && (
        <div className="bg-white rounded-xl border border-slate-200 p-6">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">Paper Trading Capital</h2>

          <div className="flex items-end gap-4">
            <div className="flex-1">
              <label className="block text-sm text-slate-500 mb-2">Current Capital</label>
              <p className="text-2xl font-bold text-slate-800">
                {formatCurrency(settings.initial_capital)}
              </p>
            </div>

            <div className="flex-1">
              <label className="block text-sm text-slate-500 mb-2">Set New Capital</label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={capitalInput}
                  onChange={(e) => setCapitalInput(e.target.value)}
                  placeholder="e.g., 1000000"
                  className="flex-1 px-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <button
                  onClick={handleSetCapital}
                  disabled={isSettingCapital || !capitalInput}
                  className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center gap-2"
                >
                  {isSettingCapital && <Loader2 className="h-4 w-4 animate-spin" />}
                  Set
                </button>
              </div>
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-100">
            <button
              onClick={handleReset}
              disabled={isResetting}
              className="flex items-center gap-2 px-4 py-2 text-slate-600 hover:bg-slate-100 rounded-lg transition-colors disabled:opacity-50"
            >
              {isResetting ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              Reset Paper Trading Account
            </button>
          </div>
        </div>
      )}

      {/* API Configuration Status */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-800 mb-4">API Configuration</h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg">
            <div
              className={clsx(
                'w-3 h-3 rounded-full',
                settings?.api_configured.zerodha ? 'bg-emerald-500' : 'bg-slate-300'
              )}
            />
            <div>
              <p className="font-medium text-slate-800">Zerodha Kite</p>
              <p className="text-sm text-slate-500">
                {settings?.api_configured.zerodha ? 'Configured' : 'Not configured'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg">
            <div
              className={clsx(
                'w-3 h-3 rounded-full',
                settings?.api_configured.firecrawl ? 'bg-emerald-500' : 'bg-slate-300'
              )}
            />
            <div>
              <p className="font-medium text-slate-800">Firecrawl</p>
              <p className="text-sm text-slate-500">
                {settings?.api_configured.firecrawl ? 'Configured' : 'Not configured'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3 p-4 bg-slate-50 rounded-lg">
            <div
              className={clsx(
                'w-3 h-3 rounded-full',
                settings?.api_configured.gemini ? 'bg-emerald-500' : 'bg-slate-300'
              )}
            />
            <div>
              <p className="font-medium text-slate-800">Gemini AI</p>
              <p className="text-sm text-slate-500">
                {settings?.api_configured.gemini ? 'Configured' : 'Not configured'}
              </p>
            </div>
          </div>
        </div>

        <p className="text-sm text-slate-400 mt-4">
          Configure API keys in the .env file on the server.
        </p>
      </div>
    </div>
  );
}
