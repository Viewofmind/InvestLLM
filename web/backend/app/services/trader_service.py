"""
Trader Service - Main trading orchestration
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from app.core.config import settings
from app.core.websocket import ConnectionManager

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    pnl: float
    pnl_pct: float

    def to_dict(self):
        return asdict(self)


@dataclass
class Trade:
    trade_id: str
    symbol: str
    action: str
    quantity: int
    price: float
    timestamp: str
    status: str
    pnl: float = 0.0

    def to_dict(self):
        return asdict(self)


class TraderService:
    """
    Main trading service that orchestrates all trading operations.

    Wraps the InvestLLM trading components and provides a clean API.
    """

    def __init__(self, ws_manager: ConnectionManager):
        self.ws_manager = ws_manager
        self.mode = TradingMode.PAPER
        self.is_running = False

        # Portfolio state
        self.initial_capital = settings.INITIAL_CAPITAL
        self.cash = self.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Risk state
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        self.daily_start_value = self.initial_capital

        # Order counter
        self._trade_counter = 1000

        # Background task
        self._update_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the trading service"""
        logger.info(f"Initializing trader in {self.mode.value} mode")
        logger.info(f"Initial capital: {self.initial_capital:,.0f}")

        # Start background update task
        self._update_task = asyncio.create_task(self._background_updates())

    async def shutdown(self):
        """Shutdown the trading service"""
        self.is_running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("Trader service shutdown complete")

    async def _background_updates(self):
        """Background task for periodic updates"""
        while True:
            try:
                # Update positions with current prices
                await self._update_positions()

                # Broadcast portfolio update
                await self.ws_manager.broadcast_portfolio_update(
                    self.get_portfolio_summary()
                )

                await asyncio.sleep(5)  # Update every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background update error: {e}")
                await asyncio.sleep(10)

    async def _update_positions(self):
        """Update positions with simulated price changes"""
        import random

        for symbol, pos in self.positions.items():
            # Simulate small price movement (paper trading)
            change = random.uniform(-0.005, 0.005)
            pos.current_price = pos.current_price * (1 + change)
            pos.pnl = (pos.current_price - pos.avg_price) * pos.quantity
            pos.pnl_pct = ((pos.current_price / pos.avg_price) - 1) * 100

    # =========================================================================
    # MODE SWITCHING
    # =========================================================================

    async def set_mode(self, mode: str) -> Dict:
        """Switch between paper and live trading"""
        if mode not in ["paper", "live"]:
            raise ValueError("Mode must be 'paper' or 'live'")

        if mode == "live" and self.mode == TradingMode.PAPER:
            # Switching to live - verify API credentials
            if not settings.ZERODHA_API_KEY:
                raise ValueError("Live trading requires Zerodha API credentials")

        old_mode = self.mode
        self.mode = TradingMode(mode)

        logger.info(f"Mode switched: {old_mode.value} -> {self.mode.value}")

        return {
            "success": True,
            "mode": self.mode.value,
            "message": f"Switched to {mode} trading"
        }

    # =========================================================================
    # TRADING OPERATIONS
    # =========================================================================

    async def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: Optional[float] = None
    ) -> Trade:
        """Place a buy or sell order"""
        action = action.upper()
        if action not in ["BUY", "SELL"]:
            raise ValueError("Action must be BUY or SELL")

        # Get current price (simulated for paper trading)
        if price is None:
            price = self._get_simulated_price(symbol)

        # Validate order
        if action == "BUY":
            order_value = price * quantity
            if order_value > self.cash:
                raise ValueError(f"Insufficient funds. Need {order_value:,.0f}, have {self.cash:,.0f}")

        elif action == "SELL":
            if symbol not in self.positions:
                raise ValueError(f"No position in {symbol}")
            if self.positions[symbol].quantity < quantity:
                raise ValueError(f"Insufficient shares. Have {self.positions[symbol].quantity}")

        # Execute order
        trade = Trade(
            trade_id=f"T{self._trade_counter}",
            symbol=symbol,
            action=action,
            quantity=quantity,
            price=price,
            timestamp=datetime.now().isoformat(),
            status="EXECUTED"
        )
        self._trade_counter += 1

        # Update positions
        if action == "BUY":
            self._process_buy(symbol, quantity, price)
        else:
            trade.pnl = self._process_sell(symbol, quantity, price)

        self.trades.append(trade)

        # Broadcast trade
        await self.ws_manager.broadcast_trade(trade.to_dict())

        logger.info(f"Order executed: {action} {quantity} {symbol} @ {price:.2f}")

        return trade

    def _process_buy(self, symbol: str, quantity: int, price: float):
        """Process buy order"""
        order_value = price * quantity
        self.cash -= order_value

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_cost = (pos.avg_price * pos.quantity) + order_value
            new_qty = pos.quantity + quantity
            pos.avg_price = total_cost / new_qty
            pos.quantity = new_qty
            pos.current_price = price
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                pnl=0,
                pnl_pct=0
            )

    def _process_sell(self, symbol: str, quantity: int, price: float) -> float:
        """Process sell order, returns realized P&L"""
        pos = self.positions[symbol]
        pnl = (price - pos.avg_price) * quantity

        self.cash += price * quantity
        pos.quantity -= quantity

        if pos.quantity <= 0:
            del self.positions[symbol]
        else:
            pos.current_price = price
            pos.pnl = (price - pos.avg_price) * pos.quantity
            pos.pnl_pct = ((price / pos.avg_price) - 1) * 100

        return pnl

    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for paper trading"""
        import random

        # Base price from symbol hash
        base = (hash(symbol) % 10000) / 10 + 100

        # Add some randomness
        return round(base * random.uniform(0.99, 1.01), 2)

    async def close_position(self, symbol: str) -> Trade:
        """Close entire position"""
        if symbol not in self.positions:
            raise ValueError(f"No position in {symbol}")

        pos = self.positions[symbol]
        return await self.place_order(symbol, "SELL", pos.quantity)

    async def close_all_positions(self) -> List[Trade]:
        """Close all positions"""
        trades = []
        symbols = list(self.positions.keys())

        for symbol in symbols:
            trade = await self.close_position(symbol)
            trades.append(trade)

        return trades

    # =========================================================================
    # PORTFOLIO DATA
    # =========================================================================

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        positions_list = [pos.to_dict() for pos in self.positions.values()]
        invested = sum(p.avg_price * p.quantity for p in self.positions.values())
        unrealized_pnl = sum(p.pnl for p in self.positions.values())
        total_value = self.cash + invested + unrealized_pnl

        # Update peak and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value

        current_drawdown = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        return {
            "mode": self.mode.value,
            "total_value": round(total_value, 2),
            "cash": round(self.cash, 2),
            "invested": round(invested, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(sum(t.pnl for t in self.trades if t.action == "SELL"), 2),
            "total_return_pct": round((total_value / self.initial_capital - 1) * 100, 2),
            "positions": positions_list,
            "positions_count": len(positions_list),
            "timestamp": datetime.now().isoformat()
        }

    def get_positions(self) -> List[Dict]:
        """Get all positions"""
        return [pos.to_dict() for pos in self.positions.values()]

    def get_trades(self, limit: int = 50) -> List[Dict]:
        """Get recent trades"""
        return [t.to_dict() for t in self.trades[-limit:]]

    # =========================================================================
    # RISK DATA
    # =========================================================================

    def get_risk_summary(self) -> Dict:
        """Get risk metrics summary"""
        total_value = self.cash + sum(
            p.avg_price * p.quantity + p.pnl
            for p in self.positions.values()
        )

        current_drawdown = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0

        # Sector exposure
        sector_map = {
            "RELIANCE": "Energy", "TCS": "IT", "INFY": "IT",
            "HDFCBANK": "Banking", "ICICIBANK": "Banking",
            # Add more mappings as needed
        }

        sector_exposure = {}
        for pos in self.positions.values():
            sector = sector_map.get(pos.symbol, "Other")
            value = pos.avg_price * pos.quantity
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value

        if total_value > 0:
            sector_exposure = {k: v / total_value for k, v in sector_exposure.items()}

        # Risk level
        if current_drawdown >= 0.15:
            risk_level = "CRITICAL"
        elif current_drawdown >= 0.10:
            risk_level = "HIGH"
        elif current_drawdown >= 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "current_drawdown_pct": round(current_drawdown * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "peak_value": round(self.peak_value, 2),
            "sector_exposure": {k: round(v * 100, 1) for k, v in sector_exposure.items()},
            "position_count": len(self.positions),
            "largest_position_pct": round(
                max((p.avg_price * p.quantity / total_value * 100 for p in self.positions.values()), default=0),
                1
            ),
            "limits": {
                "max_position_pct": settings.MAX_POSITION_PCT * 100,
                "max_sector_pct": settings.MAX_SECTOR_PCT * 100,
                "max_drawdown_pct": settings.MAX_DRAWDOWN_PCT * 100,
                "daily_loss_limit_pct": settings.DAILY_LOSS_LIMIT_PCT * 100
            }
        }

    # =========================================================================
    # RESET
    # =========================================================================

    async def reset(self) -> Dict:
        """Reset trading state (paper trading only)"""
        if self.mode == TradingMode.LIVE:
            raise ValueError("Cannot reset in live trading mode")

        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.peak_value = self.initial_capital
        self.max_drawdown = 0.0
        self.daily_start_value = self.initial_capital

        logger.info("Paper trading reset complete")

        return {
            "success": True,
            "message": "Paper trading account reset",
            "capital": self.initial_capital
        }
