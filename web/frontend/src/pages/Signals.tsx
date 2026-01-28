import { useState } from 'react';
import { useTradingStore } from '../store';
import { signalsAPI } from '../services/api';
import { Check, X, Zap, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { formatCurrency, formatPercent, formatRelativeTime } from '../utils/format';

export default function Signals() {
  const { signals, fetchSignals, fetchPositions, fetchPortfolio } = useTradingStore();
  const [executingId, setExecutingId] = useState<string | null>(null);
  const [dismissingId, setDismissingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all' | 'pending' | 'executed'>('all');

  const handleExecute = async (signalId: string) => {
    setExecutingId(signalId);
    setError(null);
    try {
      await signalsAPI.executeSignal(signalId);
      await Promise.all([fetchSignals(), fetchPositions(), fetchPortfolio()]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setExecutingId(null);
    }
  };

  const handleDismiss = async (signalId: string) => {
    setDismissingId(signalId);
    setError(null);
    try {
      await signalsAPI.dismissSignal(signalId);
      await fetchSignals();
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setDismissingId(null);
    }
  };

  const filteredSignals = signals.filter((s) => {
    if (filter === 'pending') return !s.executed;
    if (filter === 'executed') return s.executed;
    return true;
  });

  const pendingCount = signals.filter((s) => !s.executed).length;

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Trading Signals</h1>
          <p className="text-slate-500 mt-1">
            {pendingCount} pending signal{pendingCount !== 1 ? 's' : ''}
          </p>
        </div>

        {/* Filter Tabs */}
        <div className="flex gap-1 bg-slate-100 rounded-lg p-1">
          {(['all', 'pending', 'executed'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={clsx(
                'px-4 py-2 text-sm font-medium rounded-md transition-colors capitalize',
                filter === f
                  ? 'bg-white text-slate-800 shadow-sm'
                  : 'text-slate-500 hover:text-slate-700'
              )}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Signals List */}
      <div className="space-y-4">
        {filteredSignals.length === 0 ? (
          <div className="bg-white rounded-xl border border-slate-200 p-12 text-center">
            <Zap className="h-12 w-12 text-slate-300 mx-auto mb-4" />
            <p className="text-slate-500">No {filter !== 'all' ? filter : ''} signals</p>
          </div>
        ) : (
          filteredSignals.map((signal) => (
            <div
              key={signal.id}
              className={clsx(
                'bg-white rounded-xl border p-5',
                signal.executed ? 'border-slate-200 opacity-75' : 'border-slate-200'
              )}
            >
              <div className="flex items-start justify-between">
                {/* Signal Info */}
                <div className="flex items-start gap-4">
                  {/* Signal Badge */}
                  <div
                    className={clsx(
                      'px-3 py-2 rounded-lg text-center min-w-[80px]',
                      signal.signal.includes('BUY')
                        ? 'bg-emerald-100'
                        : signal.signal.includes('SELL')
                        ? 'bg-red-100'
                        : 'bg-slate-100'
                    )}
                  >
                    <p
                      className={clsx(
                        'font-bold text-sm',
                        signal.signal.includes('BUY')
                          ? 'text-emerald-700'
                          : signal.signal.includes('SELL')
                          ? 'text-red-700'
                          : 'text-slate-700'
                      )}
                    >
                      {signal.signal}
                    </p>
                    <p className="text-xs text-slate-500 mt-0.5">
                      {formatPercent(signal.strength * 100).replace('+', '')} str
                    </p>
                  </div>

                  {/* Details */}
                  <div>
                    <h3 className="font-semibold text-lg text-slate-800">{signal.symbol}</h3>
                    <p className="text-sm text-slate-500 mt-1">
                      Source: {signal.source} • Confidence: {formatPercent(signal.confidence * 100)}
                    </p>
                    <div className="flex gap-4 mt-2 text-sm">
                      {signal.price_target && (
                        <span className="text-slate-600">
                          Target: {formatCurrency(signal.price_target)}
                        </span>
                      )}
                      {signal.stop_loss && (
                        <span className="text-slate-600">
                          Stop Loss: {formatCurrency(signal.stop_loss)}
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                {/* Actions & Time */}
                <div className="text-right">
                  <p className="text-sm text-slate-400 mb-3">
                    {formatRelativeTime(signal.timestamp)}
                  </p>

                  {signal.executed ? (
                    <span className="inline-flex items-center gap-1 px-3 py-1 bg-slate-100 text-slate-600 rounded-lg text-sm">
                      <Check className="h-4 w-4" />
                      Executed
                    </span>
                  ) : (
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleExecute(signal.id)}
                        disabled={executingId === signal.id}
                        className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 flex items-center gap-2 text-sm"
                      >
                        {executingId === signal.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Check className="h-4 w-4" />
                        )}
                        Execute
                      </button>
                      <button
                        onClick={() => handleDismiss(signal.id)}
                        disabled={dismissingId === signal.id}
                        className="px-4 py-2 bg-slate-100 text-slate-600 rounded-lg hover:bg-slate-200 disabled:opacity-50 flex items-center gap-2 text-sm"
                      >
                        {dismissingId === signal.id ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <X className="h-4 w-4" />
                        )}
                        Dismiss
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
