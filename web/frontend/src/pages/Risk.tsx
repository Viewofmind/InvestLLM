import { useTradingStore } from '../store';
import { riskAPI } from '../services/api';
import { Shield, AlertTriangle, CheckCircle, Info, Bell } from 'lucide-react';
import clsx from 'clsx';
import { formatPercent, formatCurrency, formatRelativeTime } from '../utils/format';

export default function Risk() {
  const { risk, alerts, settings, fetchAlerts } = useTradingStore();

  const handleAcknowledge = async (alertId: string) => {
    try {
      await riskAPI.acknowledgeAlert(alertId);
      await fetchAlerts();
    } catch (error) {
      console.error('Failed to acknowledge alert:', error);
    }
  };

  const unacknowledgedAlerts = alerts.filter((a) => !a.acknowledged);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-800">Risk Management</h1>
        <p className="text-slate-500 mt-1">Monitor portfolio risk and alerts</p>
      </div>

      {/* Circuit Breaker Warning */}
      {risk?.circuit_breaker_active && (
        <div className="bg-red-50 border-2 border-red-300 rounded-xl p-6">
          <div className="flex items-start gap-4">
            <div className="p-3 bg-red-100 rounded-full">
              <AlertTriangle className="h-8 w-8 text-red-600" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-red-800">Circuit Breaker Active</h2>
              <p className="text-red-600 mt-1">
                All trading has been halted. Maximum drawdown limit of{' '}
                {settings?.risk_limits.max_drawdown_pct}% has been exceeded.
              </p>
              <p className="text-sm text-red-500 mt-2">
                Contact support or reset the paper trading account to resume.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Current Drawdown */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm text-slate-500">Current Drawdown</p>
            <Shield
              className={clsx(
                'h-5 w-5',
                (risk?.current_drawdown ?? 0) > 10
                  ? 'text-red-500'
                  : (risk?.current_drawdown ?? 0) > 5
                  ? 'text-amber-500'
                  : 'text-emerald-500'
              )}
            />
          </div>
          <p
            className={clsx(
              'text-2xl font-bold',
              (risk?.current_drawdown ?? 0) > 10
                ? 'text-red-600'
                : (risk?.current_drawdown ?? 0) > 5
                ? 'text-amber-600'
                : 'text-slate-800'
            )}
          >
            {formatPercent(risk?.current_drawdown ?? 0)}
          </p>
          <div className="mt-3 bg-slate-100 rounded-full h-2 overflow-hidden">
            <div
              className={clsx(
                'h-full rounded-full transition-all',
                (risk?.current_drawdown ?? 0) > 10
                  ? 'bg-red-500'
                  : (risk?.current_drawdown ?? 0) > 5
                  ? 'bg-amber-500'
                  : 'bg-emerald-500'
              )}
              style={{
                width: `${Math.min((risk?.current_drawdown ?? 0) / (settings?.risk_limits.max_drawdown_pct ?? 15) * 100, 100)}%`,
              }}
            />
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Limit: {settings?.risk_limits.max_drawdown_pct}%
          </p>
        </div>

        {/* Max Drawdown */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm text-slate-500">Max Drawdown</p>
            <Info className="h-5 w-5 text-slate-400" />
          </div>
          <p className="text-2xl font-bold text-slate-800">
            {formatPercent(risk?.max_drawdown ?? 0)}
          </p>
          <p className="text-xs text-slate-400 mt-3">Highest drawdown recorded</p>
        </div>

        {/* Daily P&L */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm text-slate-500">Daily P&L</p>
            {(risk?.daily_pnl ?? 0) >= 0 ? (
              <CheckCircle className="h-5 w-5 text-emerald-500" />
            ) : (
              <AlertTriangle className="h-5 w-5 text-red-500" />
            )}
          </div>
          <p
            className={clsx(
              'text-2xl font-bold',
              (risk?.daily_pnl ?? 0) >= 0 ? 'text-emerald-600' : 'text-red-600'
            )}
          >
            {formatCurrency(risk?.daily_pnl ?? 0)}
          </p>
          <p className="text-xs text-slate-400 mt-3">
            Limit: -{settings?.risk_limits.daily_loss_limit_pct}%
          </p>
        </div>

        {/* Portfolio Exposure */}
        <div className="bg-white rounded-xl border border-slate-200 p-5">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm text-slate-500">Portfolio Exposure</p>
            <Shield className="h-5 w-5 text-primary-500" />
          </div>
          <p className="text-2xl font-bold text-slate-800">
            {formatPercent(risk?.exposure_pct ?? 0)}
          </p>
          <p className="text-xs text-slate-400 mt-3">
            {risk?.position_count ?? 0} active positions
          </p>
        </div>
      </div>

      {/* Risk Limits */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <h2 className="text-lg font-semibold text-slate-800 mb-4">Risk Limits</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div>
            <p className="text-sm text-slate-500">Max Position Size</p>
            <p className="text-xl font-semibold text-slate-800 mt-1">
              {settings?.risk_limits.max_position_pct}%
            </p>
            <p className="text-xs text-slate-400">per position</p>
          </div>
          <div>
            <p className="text-sm text-slate-500">Max Sector Exposure</p>
            <p className="text-xl font-semibold text-slate-800 mt-1">
              {settings?.risk_limits.max_sector_pct}%
            </p>
            <p className="text-xs text-slate-400">per sector</p>
          </div>
          <div>
            <p className="text-sm text-slate-500">Max Drawdown</p>
            <p className="text-xl font-semibold text-slate-800 mt-1">
              {settings?.risk_limits.max_drawdown_pct}%
            </p>
            <p className="text-xs text-slate-400">triggers circuit breaker</p>
          </div>
          <div>
            <p className="text-sm text-slate-500">Daily Loss Limit</p>
            <p className="text-xl font-semibold text-slate-800 mt-1">
              {settings?.risk_limits.daily_loss_limit_pct}%
            </p>
            <p className="text-xs text-slate-400">halts trading for the day</p>
          </div>
        </div>
      </div>

      {/* Alerts */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-slate-800">Risk Alerts</h2>
          {unacknowledgedAlerts.length > 0 && (
            <span className="flex items-center gap-1 px-2 py-1 bg-amber-100 text-amber-700 rounded-md text-sm">
              <Bell className="h-4 w-4" />
              {unacknowledgedAlerts.length} unacknowledged
            </span>
          )}
        </div>

        {alerts.length === 0 ? (
          <div className="text-center py-8">
            <CheckCircle className="h-12 w-12 text-emerald-300 mx-auto mb-3" />
            <p className="text-slate-500">No risk alerts</p>
            <p className="text-sm text-slate-400">All risk metrics are within limits</p>
          </div>
        ) : (
          <div className="space-y-3">
            {alerts.map((alert) => (
              <div
                key={alert.id}
                className={clsx(
                  'flex items-start justify-between p-4 rounded-lg border',
                  alert.acknowledged
                    ? 'bg-slate-50 border-slate-200 opacity-60'
                    : alert.level === 'critical'
                    ? 'bg-red-50 border-red-200'
                    : 'bg-amber-50 border-amber-200'
                )}
              >
                <div className="flex items-start gap-3">
                  <AlertTriangle
                    className={clsx(
                      'h-5 w-5 mt-0.5',
                      alert.acknowledged
                        ? 'text-slate-400'
                        : alert.level === 'critical'
                        ? 'text-red-500'
                        : 'text-amber-500'
                    )}
                  />
                  <div>
                    <p
                      className={clsx(
                        'font-medium',
                        alert.acknowledged
                          ? 'text-slate-600'
                          : alert.level === 'critical'
                          ? 'text-red-800'
                          : 'text-amber-800'
                      )}
                    >
                      {alert.message}
                    </p>
                    <p className="text-sm text-slate-500 mt-1">
                      {formatRelativeTime(alert.timestamp)}
                    </p>
                  </div>
                </div>
                {!alert.acknowledged && (
                  <button
                    onClick={() => handleAcknowledge(alert.id)}
                    className="px-3 py-1.5 text-sm text-slate-600 hover:bg-slate-200 rounded-md transition-colors"
                  >
                    Acknowledge
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
