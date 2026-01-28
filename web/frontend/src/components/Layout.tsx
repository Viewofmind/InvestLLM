import { Link, useLocation } from 'react-router-dom';
import { useTradingStore } from '../store';
import {
  LayoutDashboard,
  Briefcase,
  Zap,
  Shield,
  Settings,
  Activity,
  Wifi,
  WifiOff,
} from 'lucide-react';
import clsx from 'clsx';

interface LayoutProps {
  children: React.ReactNode;
  isConnected: boolean;
}

const navItems = [
  { path: '/', label: 'Dashboard', icon: LayoutDashboard },
  { path: '/positions', label: 'Positions', icon: Briefcase },
  { path: '/signals', label: 'Signals', icon: Zap },
  { path: '/risk', label: 'Risk', icon: Shield },
  { path: '/settings', label: 'Settings', icon: Settings },
];

export default function Layout({ children, isConnected }: LayoutProps) {
  const location = useLocation();
  const { settings, alerts } = useTradingStore();

  const unacknowledgedAlerts = alerts.filter((a) => !a.acknowledged).length;
  const isLiveMode = settings?.mode === 'live';
  const isRunning = settings?.is_running;

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <Activity className="h-8 w-8 text-primary-600" />
              <span className="font-bold text-xl text-slate-800">InvestLLM</span>
              <span
                className={clsx(
                  'px-2 py-0.5 text-xs font-medium rounded-full',
                  isLiveMode
                    ? 'bg-red-100 text-red-700'
                    : 'bg-emerald-100 text-emerald-700'
                )}
              >
                {isLiveMode ? 'LIVE' : 'PAPER'}
              </span>
            </div>

            {/* Status Indicators */}
            <div className="flex items-center gap-4">
              {/* Trading Status */}
              {isRunning && (
                <div className="flex items-center gap-2 text-sm">
                  <span className="w-2 h-2 bg-emerald-500 rounded-full pulse-green" />
                  <span className="text-emerald-600 font-medium">Trading Active</span>
                </div>
              )}

              {/* Alerts Badge */}
              {unacknowledgedAlerts > 0 && (
                <Link
                  to="/risk"
                  className="flex items-center gap-1 px-2 py-1 bg-amber-100 text-amber-700 rounded-md text-sm font-medium"
                >
                  <Shield className="h-4 w-4" />
                  {unacknowledgedAlerts} Alert{unacknowledgedAlerts > 1 ? 's' : ''}
                </Link>
              )}

              {/* Connection Status */}
              <div
                className={clsx(
                  'flex items-center gap-1 px-2 py-1 rounded-md text-sm',
                  isConnected
                    ? 'bg-emerald-100 text-emerald-700'
                    : 'bg-red-100 text-red-700'
                )}
              >
                {isConnected ? (
                  <>
                    <Wifi className="h-4 w-4" />
                    <span>Live</span>
                  </>
                ) : (
                  <>
                    <WifiOff className="h-4 w-4" />
                    <span>Offline</span>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar Navigation */}
        <nav className="w-64 bg-white border-r border-slate-200 min-h-[calc(100vh-4rem)] p-4">
          <ul className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;

              return (
                <li key={item.path}>
                  <Link
                    to={item.path}
                    className={clsx(
                      'flex items-center gap-3 px-4 py-2.5 rounded-lg text-sm font-medium transition-colors',
                      isActive
                        ? 'bg-primary-50 text-primary-700'
                        : 'text-slate-600 hover:bg-slate-100'
                    )}
                  >
                    <Icon className="h-5 w-5" />
                    {item.label}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-6">{children}</main>
      </div>
    </div>
  );
}
