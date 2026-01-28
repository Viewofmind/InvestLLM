"""
InvestLLM Trading Module
========================
Live and paper trading components.

Components:
- BaseBacktester: Backtesting engine
- KiteAPI: Zerodha Kite Connect integration
- LiveTrader: Real-time trading engine
- RiskManager: Portfolio risk management
"""

from .backtester import BaseBacktester

from .kite_api import (
    KiteAPI,
    TransactionType,
    OrderType,
    ProductType,
    Quote,
    Order,
    Position
)

from .live_trader import (
    LiveTrader,
    TradingMode,
    SignalType,
    TradingSignal,
    TradeRecord
)

from .risk_manager import (
    RiskManager,
    RiskLevel,
    AlertType,
    RiskAlert,
    PortfolioRisk
)

__all__ = [
    # Backtester
    "BaseBacktester",
    # Kite API
    "KiteAPI",
    "TransactionType",
    "OrderType",
    "ProductType",
    "Quote",
    "Order",
    "Position",
    # Live Trader
    "LiveTrader",
    "TradingMode",
    "SignalType",
    "TradingSignal",
    "TradeRecord",
    # Risk Manager
    "RiskManager",
    "RiskLevel",
    "AlertType",
    "RiskAlert",
    "PortfolioRisk"
]
