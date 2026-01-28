"""
Live Trader for InvestLLM
=========================
Real-time trading engine with paper trading mode.

Features:
- Paper trading (mock orders) for testing
- Live trading via Kite Connect
- Signal-based order execution
- Position management
- Real-time P&L tracking
- Risk limit enforcement

Components:
- KiteAPI: Market data and order execution
- SmartExitManager: Exit strategy management
- RiskManager: Position sizing and limits
"""

import os
import json
import asyncio
import logging
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import schedule

# Local imports
from investllm.trading.kite_api import (
    KiteAPI, TransactionType, OrderType, ProductType, Quote, Order, Position
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode"""
    PAPER = "paper"
    LIVE = "live"


class SignalType(Enum):
    """Trading signal types"""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class TradingSignal:
    """Trading signal from models"""
    symbol: str
    signal: SignalType
    strength: float  # 0.0 to 1.0
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    source: str = "ensemble"  # model source
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class TradeRecord:
    """Record of executed trade"""
    trade_id: str
    symbol: str
    action: str  # BUY/SELL
    quantity: int
    price: float
    timestamp: datetime
    signal_source: str
    order_id: str = ""
    status: str = "PENDING"  # PENDING, EXECUTED, FAILED
    pnl: float = 0.0
    notes: str = ""


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time"""
    timestamp: datetime
    total_value: float
    cash: float
    invested: float
    unrealized_pnl: float
    realized_pnl: float
    positions: List[Dict]
    max_drawdown: float


class LiveTrader:
    """
    Real-time trading engine for InvestLLM.

    Supports both paper trading and live trading modes.
    Integrates with model signals and risk management.
    """

    # Market hours (IST)
    MARKET_OPEN = dtime(9, 15)
    MARKET_CLOSE = dtime(15, 30)

    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        initial_capital: float = 100000,
        max_position_size: float = 0.10,  # 10% per position
        max_positions: int = 10,
        stop_loss_pct: float = 0.15,  # 15% stop loss
        profit_target_pct: float = 0.50,  # 50% profit target
        data_dir: str = "data/trading"
    ):
        """
        Initialize live trader.

        Args:
            mode: PAPER or LIVE trading
            initial_capital: Starting capital
            max_position_size: Max % of capital per position
            max_positions: Maximum concurrent positions
            stop_loss_pct: Default stop loss percentage
            profit_target_pct: Default profit target percentage
            data_dir: Directory for trade logs
        """
        self.mode = mode
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct

        # Data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Kite API
        self.kite = KiteAPI(paper_trading=(mode == TradingMode.PAPER))
        if mode == TradingMode.PAPER:
            self.kite.reset_paper_trading(initial_capital)

        # Trade tracking
        self.trades: List[TradeRecord] = []
        self.pending_signals: List[TradingSignal] = []
        self.portfolio_history: List[PortfolioSnapshot] = []

        # Peak value for drawdown calculation
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0

        # Running state
        self._running = False
        self._trade_id_counter = 1000

        # Callbacks
        self._on_trade_callbacks: List[Callable] = []
        self._on_signal_callbacks: List[Callable] = []

        logger.info(f"LiveTrader initialized in {mode.value} mode")
        logger.info(f"Capital: {initial_capital:,.0f} | Max position: {max_position_size:.0%}")

    # =========================================================================
    # SIGNAL HANDLING
    # =========================================================================

    def add_signal(self, signal: TradingSignal):
        """
        Add a trading signal for processing.

        Args:
            signal: TradingSignal object
        """
        self.pending_signals.append(signal)
        logger.info(f"Signal added: {signal.signal.value} {signal.symbol} (strength: {signal.strength:.2f})")

        for callback in self._on_signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def add_signals_batch(self, signals: List[TradingSignal]):
        """Add multiple signals"""
        for signal in signals:
            self.add_signal(signal)

    async def process_signals(self):
        """Process pending signals and execute trades"""
        if not self.pending_signals:
            return

        if not self._is_market_hours():
            logger.info("Market closed - signals queued for next session")
            return

        # Sort by signal strength (strongest first)
        signals = sorted(
            self.pending_signals,
            key=lambda s: s.strength,
            reverse=True
        )

        for signal in signals:
            try:
                await self._process_single_signal(signal)
            except Exception as e:
                logger.error(f"Signal processing error for {signal.symbol}: {e}")

        # Clear processed signals
        self.pending_signals = []

    async def _process_single_signal(self, signal: TradingSignal):
        """Process a single trading signal"""
        positions = self.kite.get_positions()
        position_symbols = {p.symbol for p in positions}

        if signal.signal in [SignalType.STRONG_BUY, SignalType.BUY]:
            # Check if already in position
            if signal.symbol in position_symbols:
                logger.info(f"Already in position: {signal.symbol}")
                return

            # Check position limits
            if len(positions) >= self.max_positions:
                logger.info(f"Max positions reached ({self.max_positions})")
                return

            # Calculate position size
            quote = self.kite.get_quote(signal.symbol)
            position_value = self._calculate_position_size(signal)
            quantity = int(position_value / quote.last_price)

            if quantity < 1:
                logger.info(f"Position too small for {signal.symbol}")
                return

            # Execute buy
            await self._execute_buy(signal, quantity, quote.last_price)

        elif signal.signal in [SignalType.STRONG_SELL, SignalType.SELL]:
            # Check if in position
            if signal.symbol not in position_symbols:
                logger.info(f"No position to sell: {signal.symbol}")
                return

            # Get position
            pos = next(p for p in positions if p.symbol == signal.symbol)

            # Execute sell
            await self._execute_sell(signal, pos)

    def _calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on signal and risk rules"""
        margins = self.kite.get_margins()
        available_cash = margins.get("available_cash", 0)
        portfolio_value = margins.get("portfolio_value", self.initial_capital)

        # Base position size
        max_position = portfolio_value * self.max_position_size

        # Adjust by signal strength
        if signal.signal == SignalType.STRONG_BUY:
            size_multiplier = 1.0
        elif signal.signal == SignalType.BUY:
            size_multiplier = 0.7
        else:
            size_multiplier = 0.5

        position_size = max_position * size_multiplier * signal.strength

        # Don't exceed available cash
        position_size = min(position_size, available_cash * 0.95)

        return position_size

    async def _execute_buy(
        self,
        signal: TradingSignal,
        quantity: int,
        price: float
    ):
        """Execute a buy order"""
        order = self.kite.place_order(
            symbol=signal.symbol,
            transaction_type=TransactionType.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
            product=ProductType.CNC
        )

        # Record trade
        trade = TradeRecord(
            trade_id=f"T{self._trade_id_counter}",
            symbol=signal.symbol,
            action="BUY",
            quantity=quantity,
            price=price,
            timestamp=datetime.now(),
            signal_source=signal.source,
            order_id=order.order_id,
            status="EXECUTED" if order.status == "COMPLETE" else "PENDING"
        )
        self._trade_id_counter += 1
        self.trades.append(trade)

        logger.info(
            f"BUY EXECUTED: {quantity} {signal.symbol} @ {price:.2f} "
            f"(Value: {quantity * price:,.0f})"
        )

        self._notify_trade(trade)

    async def _execute_sell(self, signal: TradingSignal, position: Position):
        """Execute a sell order"""
        order = self.kite.place_order(
            symbol=signal.symbol,
            transaction_type=TransactionType.SELL,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            product=ProductType.CNC
        )

        # Record trade
        trade = TradeRecord(
            trade_id=f"T{self._trade_id_counter}",
            symbol=signal.symbol,
            action="SELL",
            quantity=position.quantity,
            price=position.last_price,
            timestamp=datetime.now(),
            signal_source=signal.source,
            order_id=order.order_id,
            status="EXECUTED" if order.status == "COMPLETE" else "PENDING",
            pnl=position.pnl
        )
        self._trade_id_counter += 1
        self.trades.append(trade)

        self.realized_pnl += position.pnl

        logger.info(
            f"SELL EXECUTED: {position.quantity} {signal.symbol} @ {position.last_price:.2f} "
            f"(P&L: {position.pnl:+,.0f})"
        )

        self._notify_trade(trade)

    # =========================================================================
    # POSITION MONITORING
    # =========================================================================

    async def check_exits(self):
        """Check all positions for exit conditions"""
        positions = self.kite.get_positions()

        for position in positions:
            should_exit, reason = self._check_exit_conditions(position)

            if should_exit:
                signal = TradingSignal(
                    symbol=position.symbol,
                    signal=SignalType.SELL,
                    strength=1.0,
                    source=f"exit_{reason}"
                )
                await self._execute_sell(signal, position)

    def _check_exit_conditions(self, position: Position) -> tuple:
        """
        Check if position should be exited.

        Returns:
            (should_exit: bool, reason: str)
        """
        pnl_pct = position.pnl_percent / 100

        # Stop loss hit
        if pnl_pct <= -self.stop_loss_pct:
            return True, "stop_loss"

        # Profit target hit
        if pnl_pct >= self.profit_target_pct:
            return True, "profit_target"

        return False, ""

    async def update_portfolio(self):
        """Update portfolio state and check for alerts"""
        positions = self.kite.get_positions()
        margins = self.kite.get_margins()

        total_value = margins.get("portfolio_value", self.initial_capital)
        cash = margins.get("available_cash", 0)
        invested = sum(p.average_price * p.quantity for p in positions)
        unrealized_pnl = sum(p.pnl for p in positions)

        # Update peak and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value

        current_drawdown = (self.peak_value - total_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Record snapshot
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(),
            total_value=total_value,
            cash=cash,
            invested=invested,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self.realized_pnl,
            positions=[asdict(p) for p in positions],
            max_drawdown=self.max_drawdown
        )
        self.portfolio_history.append(snapshot)

        # Drawdown alert
        if current_drawdown > 0.10:  # 10% drawdown
            logger.warning(f"DRAWDOWN ALERT: {current_drawdown:.1%}")

    # =========================================================================
    # TRADING LOOP
    # =========================================================================

    async def start(self, check_interval: int = 60):
        """
        Start the trading loop.

        Args:
            check_interval: Seconds between checks
        """
        self._running = True
        logger.info("Trading loop started")

        while self._running:
            try:
                if self._is_market_hours():
                    # Process pending signals
                    await self.process_signals()

                    # Check exit conditions
                    await self.check_exits()

                    # Update portfolio state
                    await self.update_portfolio()

                    # Log status
                    self._log_status()

                await asyncio.sleep(check_interval)

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(10)

    def stop(self):
        """Stop the trading loop"""
        self._running = False
        logger.info("Trading loop stopped")
        self._save_session()

    def _is_market_hours(self) -> bool:
        """Check if within market hours"""
        now = datetime.now().time()
        return self.MARKET_OPEN <= now <= self.MARKET_CLOSE

    def _log_status(self):
        """Log current portfolio status"""
        margins = self.kite.get_margins()
        positions = self.kite.get_positions()

        logger.info(
            f"Portfolio: {margins.get('portfolio_value', 0):,.0f} | "
            f"Cash: {margins.get('available_cash', 0):,.0f} | "
            f"Positions: {len(positions)} | "
            f"MaxDD: {self.max_drawdown:.1%}"
        )

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def on_trade(self, callback: Callable):
        """Register trade callback"""
        self._on_trade_callbacks.append(callback)

    def on_signal(self, callback: Callable):
        """Register signal callback"""
        self._on_signal_callbacks.append(callback)

    def _notify_trade(self, trade: TradeRecord):
        """Notify trade callbacks"""
        for callback in self._on_trade_callbacks:
            try:
                callback(trade)
            except Exception as e:
                logger.error(f"Trade callback error: {e}")

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_summary(self) -> Dict:
        """Get trading session summary"""
        positions = self.kite.get_positions()
        margins = self.kite.get_margins()

        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.action == "SELL" and t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.action == "SELL" and t.pnl < 0])

        return {
            "mode": self.mode.value,
            "portfolio_value": margins.get("portfolio_value", 0),
            "cash": margins.get("available_cash", 0),
            "positions": len(positions),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": winning_trades / max(winning_trades + losing_trades, 1),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": sum(p.pnl for p in positions),
            "max_drawdown": self.max_drawdown,
            "return_pct": (margins.get("portfolio_value", 0) / self.initial_capital - 1) * 100
        }

    def get_positions_report(self) -> List[Dict]:
        """Get current positions report"""
        positions = self.kite.get_positions()
        return [
            {
                "symbol": p.symbol,
                "quantity": p.quantity,
                "avg_price": p.average_price,
                "current_price": p.last_price,
                "pnl": p.pnl,
                "pnl_pct": p.pnl_percent
            }
            for p in positions
        ]

    def get_trades_report(self) -> List[Dict]:
        """Get all trades report"""
        return [asdict(t) for t in self.trades]

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    def _save_session(self):
        """Save session data to disk"""
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "mode": self.mode.value,
            "summary": self.get_summary(),
            "trades": self.get_trades_report(),
            "positions": self.get_positions_report(),
            "portfolio_history": [
                {
                    "timestamp": s.timestamp.isoformat(),
                    "total_value": s.total_value,
                    "max_drawdown": s.max_drawdown
                }
                for s in self.portfolio_history[-1000:]  # Last 1000 snapshots
            ]
        }

        filename = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.data_dir / filename

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)

        logger.info(f"Session saved to {filepath}")

    def load_session(self, filepath: str):
        """Load previous session data"""
        with open(filepath) as f:
            data = json.load(f)

        logger.info(f"Loaded session from {filepath}")
        return data


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

async def main():
    """Example usage"""
    # Create trader in paper mode
    trader = LiveTrader(
        mode=TradingMode.PAPER,
        initial_capital=100000,
        max_position_size=0.10,
        stop_loss_pct=0.15,
        profit_target_pct=0.50
    )

    # Register callbacks
    def on_trade(trade):
        print(f"Trade: {trade.action} {trade.quantity} {trade.symbol} @ {trade.price}")

    trader.on_trade(on_trade)

    # Add some test signals
    signals = [
        TradingSignal(
            symbol="RELIANCE",
            signal=SignalType.STRONG_BUY,
            strength=0.85,
            source="ensemble"
        ),
        TradingSignal(
            symbol="TCS",
            signal=SignalType.BUY,
            strength=0.72,
            source="sentiment"
        )
    ]

    trader.add_signals_batch(signals)

    # Process signals (outside market hours for demo)
    trader.MARKET_OPEN = dtime(0, 0)  # Allow processing
    trader.MARKET_CLOSE = dtime(23, 59)

    await trader.process_signals()
    await trader.update_portfolio()

    # Print summary
    print("\n" + "=" * 60)
    print("TRADING SESSION SUMMARY")
    print("=" * 60)

    summary = trader.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nPositions:")
    for pos in trader.get_positions_report():
        print(f"  {pos['symbol']}: {pos['quantity']} @ {pos['avg_price']:.2f}")

    # Save session
    trader._save_session()


if __name__ == "__main__":
    asyncio.run(main())
