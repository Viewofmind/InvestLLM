import { useState } from 'react';
import { useTradingStore } from '../store';
import { tradingAPI } from '../services/api';
import { X, TrendingUp, TrendingDown, Loader2 } from 'lucide-react';
import clsx from 'clsx';
import { formatCurrency, formatPercent, formatNumber, formatDateTime } from '../utils/format';

export default function Positions() {
  const { positions, fetchPositions, fetchPortfolio } = useTradingStore();
  const [closingSymbol, setClosingSymbol] = useState<string | null>(null);
  const [closingAll, setClosingAll] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleClosePosition = async (symbol: string) => {
    setClosingSymbol(symbol);
    setError(null);
    try {
      await tradingAPI.closePosition(symbol);
      await Promise.all([fetchPositions(), fetchPortfolio()]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setClosingSymbol(null);
    }
  };

  const handleCloseAll = async () => {
    if (!window.confirm('Are you sure you want to close all positions?')) return;

    setClosingAll(true);
    setError(null);
    try {
      await tradingAPI.closeAllPositions();
      await Promise.all([fetchPositions(), fetchPortfolio()]);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setClosingAll(false);
    }
  };

  const totalPnl = positions.reduce((sum, pos) => sum + pos.pnl, 0);
  const totalValue = positions.reduce((sum, pos) => sum + pos.current_price * pos.quantity, 0);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-slate-800">Positions</h1>
          <p className="text-slate-500 mt-1">{positions.length} open positions</p>
        </div>
        {positions.length > 0 && (
          <button
            onClick={handleCloseAll}
            disabled={closingAll}
            className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 flex items-center gap-2"
          >
            {closingAll && <Loader2 className="h-4 w-4 animate-spin" />}
            Close All
          </button>
        )}
      </div>

      {/* Error Alert */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <p className="text-sm text-slate-500">Total Value</p>
          <p className="text-2xl font-bold text-slate-800">{formatCurrency(totalValue)}</p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <p className="text-sm text-slate-500">Unrealized P&L</p>
          <p
            className={clsx(
              'text-2xl font-bold',
              totalPnl >= 0 ? 'text-emerald-600' : 'text-red-600'
            )}
          >
            {formatCurrency(totalPnl)}
          </p>
        </div>
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <p className="text-sm text-slate-500">Positions</p>
          <p className="text-2xl font-bold text-slate-800">{positions.length}</p>
        </div>
      </div>

      {/* Positions Table */}
      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
        {positions.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-slate-500">No open positions</p>
            <p className="text-sm text-slate-400 mt-1">Execute signals to open positions</p>
          </div>
        ) : (
          <table className="w-full">
            <thead className="bg-slate-50">
              <tr className="text-left text-sm text-slate-500">
                <th className="px-6 py-4 font-medium">Symbol</th>
                <th className="px-6 py-4 font-medium">Side</th>
                <th className="px-6 py-4 font-medium">Quantity</th>
                <th className="px-6 py-4 font-medium">Avg Price</th>
                <th className="px-6 py-4 font-medium">Current</th>
                <th className="px-6 py-4 font-medium">P&L</th>
                <th className="px-6 py-4 font-medium">Entry Time</th>
                <th className="px-6 py-4 font-medium"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {positions.map((pos) => (
                <tr key={pos.symbol} className="hover:bg-slate-50">
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-2">
                      {pos.pnl >= 0 ? (
                        <TrendingUp className="h-4 w-4 text-emerald-500" />
                      ) : (
                        <TrendingDown className="h-4 w-4 text-red-500" />
                      )}
                      <span className="font-medium text-slate-800">{pos.symbol}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span
                      className={clsx(
                        'px-2 py-1 text-xs font-medium rounded',
                        pos.side === 'BUY'
                          ? 'bg-emerald-100 text-emerald-700'
                          : 'bg-red-100 text-red-700'
                      )}
                    >
                      {pos.side}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-slate-600">{formatNumber(pos.quantity)}</td>
                  <td className="px-6 py-4 text-slate-600">{formatCurrency(pos.avg_price)}</td>
                  <td className="px-6 py-4 text-slate-600">{formatCurrency(pos.current_price)}</td>
                  <td className="px-6 py-4">
                    <span
                      className={clsx(
                        'font-medium',
                        pos.pnl >= 0 ? 'text-emerald-600' : 'text-red-600'
                      )}
                    >
                      {formatCurrency(pos.pnl)}
                      <span className="text-xs ml-1">({formatPercent(pos.pnl_pct)})</span>
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-slate-500">
                    {formatDateTime(pos.entry_time)}
                  </td>
                  <td className="px-6 py-4">
                    <button
                      onClick={() => handleClosePosition(pos.symbol)}
                      disabled={closingSymbol === pos.symbol}
                      className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                      title="Close position"
                    >
                      {closingSymbol === pos.symbol ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <X className="h-4 w-4" />
                      )}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
