import { useTradingStore } from '../store';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  PieChart,
  BarChart3,
  AlertTriangle,
} from 'lucide-react';
import clsx from 'clsx';
import { formatCurrency, formatPercent, formatNumber } from '../utils/format';

export default function Dashboard() {
  const { portfolio, positions, signals, risk, settings, isLoading } = useTradingStore();

  if (isLoading && !portfolio) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  const pendingSignals = signals.filter((s) => !s.executed);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Dashboard</h1>
        <p className="text-slate-500 mt-1">
          {settings?.mode === 'paper' ? 'Paper Trading' : 'Live Trading'} Overview
        </p>
      </div>

      {/* Circuit Breaker Warning */}
      {risk?.circuit_breaker_active && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
          <AlertTriangle className="h-6 w-6 text-red-600" />
          <div>
            <p className="font-semibold text-red-800">Circuit Breaker Active</p>
            <p className="text-sm text-red-600">
              Trading has been halted due to exceeding maximum drawdown limit.
            </p>
          </div>
        </div>
      )}

      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Value */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary-100 rounded-lg">
              <DollarSign className="h-5 w-5 text-primary-600" />
            </div>
            <div>
              <p className="text-sm text-slate-500">Total Value</p>
              <p className="text-xl font-bold text-slate-800">
                {formatCurrency(portfolio?.total_value ?? 0)}
              </p>
            </div>
          </div>
        </div>

        {/* Total P&L */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center gap-3">
            <div
              className={clsx(
                'p-2 rounded-lg',
                (portfolio?.total_pnl ?? 0) >= 0 ? 'bg-emerald-100' : 'bg-red-100'
              )}
            >
              {(portfolio?.total_pnl ?? 0) >= 0 ? (
                <TrendingUp className="h-5 w-5 text-emerald-600" />
              ) : (
                <TrendingDown className="h-5 w-5 text-red-600" />
              )}
            </div>
            <div>
              <p className="text-sm text-slate-500">Total P&L</p>
              <p
                className={clsx(
                  'text-xl font-bold',
                  (portfolio?.total_pnl ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-600'
                )}
              >
                {formatCurrency(portfolio?.total_pnl ?? 0)}
                <span className="text-sm ml-1">
                  ({formatPercent(portfolio?.total_pnl_pct ?? 0)})
                </span>
              </p>
            </div>
          </div>
        </div>

        {/* Day P&L */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center gap-3">
            <div
              className={clsx(
                'p-2 rounded-lg',
                (portfolio?.day_pnl ?? 0) >= 0 ? 'bg-emerald-100' : 'bg-red-100'
              )}
            >
              <BarChart3
                className={clsx(
                  'h-5 w-5',
                  (portfolio?.day_pnl ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-600'
                )}
              />
            </div>
            <div>
              <p className="text-sm text-slate-500">Today's P&L</p>
              <p
                className={clsx(
                  'text-xl font-bold',
                  (portfolio?.day_pnl ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-600'
                )}
              >
                {formatCurrency(portfolio?.day_pnl ?? 0)}
                <span className="text-sm ml-1">
                  ({formatPercent(portfolio?.day_pnl_pct ?? 0)})
                </span>
              </p>
            </div>
          </div>
        </div>

        {/* Cash Available */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-slate-100 rounded-lg">
              <PieChart className="h-5 w-5 text-slate-600" />
            </div>
            <div>
              <p className="text-sm text-slate-500">Cash Available</p>
              <p className="text-xl font-bold text-slate-800">
                {formatCurrency(portfolio?.cash ?? 0)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Positions Overview */}
        <div className="lg:col-span-2 bg-white rounded-xl border border-slate-200 p-5">
          <h2 className="text-lg font-semibold text-slate-800 mb-4">Open Positions</h2>
          {positions.length === 0 ? (
            <p className="text-slate-500 text-center py-8">No open positions</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-left text-sm text-slate-500 border-b border-slate-100">
                    <th className="pb-3 font-medium">Symbol</th>
                    <th className="pb-3 font-medium">Qty</th>
                    <th className="pb-3 font-medium">Avg Price</th>
                    <th className="pb-3 font-medium">Current</th>
                    <th className="pb-3 font-medium text-right">P&L</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  {positions.slice(0, 5).map((pos) => (
                    <tr key={pos.symbol} className="text-sm">
                      <td className="py-3 font-medium text-slate-800">{pos.symbol}</td>
                      <td className="py-3 text-slate-600">{formatNumber(pos.quantity)}</td>
                      <td className="py-3 text-slate-600">{formatCurrency(pos.avg_price)}</td>
                      <td className="py-3 text-slate-600">{formatCurrency(pos.current_price)}</td>
                      <td
                        className={clsx(
                          'py-3 text-right font-medium',
                          pos.pnl >= 0 ? 'text-emerald-600' : 'text-red-600'
                        )}
                      >
                        {formatCurrency(pos.pnl)}
                        <span className="text-xs ml-1">({formatPercent(pos.pnl_pct)})</span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {positions.length > 5 && (
                <p className="text-sm text-slate-500 mt-3 text-center">
                  +{positions.length - 5} more positions
                </p>
              )}
            </div>
          )}
        </div>

        {/* Pending Signals & Risk Summary */}
        <div className="space-y-6">
          {/* Pending Signals */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">Pending Signals</h2>
            {pendingSignals.length === 0 ? (
              <p className="text-slate-500 text-center py-4">No pending signals</p>
            ) : (
              <div className="space-y-3">
                {pendingSignals.slice(0, 4).map((signal) => (
                  <div
                    key={signal.id}
                    className="flex items-center justify-between p-3 bg-slate-50 rounded-lg"
                  >
                    <div>
                      <p className="font-medium text-slate-800">{signal.symbol}</p>
                      <p className="text-xs text-slate-500">{signal.source}</p>
                    </div>
                    <span
                      className={clsx(
                        'px-2 py-1 text-xs font-medium rounded',
                        signal.signal.includes('BUY')
                          ? 'bg-emerald-100 text-emerald-700'
                          : signal.signal.includes('SELL')
                          ? 'bg-red-100 text-red-700'
                          : 'bg-slate-100 text-slate-700'
                      )}
                    >
                      {signal.signal}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Risk Summary */}
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h2 className="text-lg font-semibold text-slate-800 mb-4">Risk Metrics</h2>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-500">Current Drawdown</span>
                <span
                  className={clsx(
                    'font-medium',
                    (risk?.current_drawdown ?? 0) > 10
                      ? 'text-red-600'
                      : (risk?.current_drawdown ?? 0) > 5
                      ? 'text-amber-600'
                      : 'text-slate-800'
                  )}
                >
                  {formatPercent(risk?.current_drawdown ?? 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-500">Max Drawdown</span>
                <span className="font-medium text-slate-800">
                  {formatPercent(risk?.max_drawdown ?? 0)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-500">Positions</span>
                <span className="font-medium text-slate-800">{risk?.position_count ?? 0}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-slate-500">Exposure</span>
                <span className="font-medium text-slate-800">
                  {formatPercent(risk?.exposure_pct ?? 0)}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
