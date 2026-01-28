"""
Kite Connect API Integration for InvestLLM
==========================================
Provides real-time market data and order execution via Zerodha Kite.

Features:
- Real-time quotes and OHLC data
- Order placement (BUY/SELL)
- Position and holdings management
- Paper trading mode for testing

Setup:
1. Get API Key from https://developers.kite.trade/
2. Set ZERODHA_API_KEY and ZERODHA_API_SECRET in .env
3. Complete login flow to get access_token
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass, field
from enum import Enum
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by Kite"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class TransactionType(Enum):
    """Transaction types"""
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """Product types for positions"""
    CNC = "CNC"      # Cash and Carry (delivery)
    MIS = "MIS"      # Margin Intraday Square-off
    NRML = "NRML"    # Normal (F&O)


@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    last_price: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime
    change: float = 0.0
    change_percent: float = 0.0


@dataclass
class Order:
    """Order details"""
    order_id: str
    symbol: str
    transaction_type: str
    quantity: int
    price: float
    status: str
    order_type: str
    timestamp: datetime
    filled_quantity: int = 0
    average_price: float = 0.0


@dataclass
class Position:
    """Position details"""
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percent: float
    product: str


class KiteAPI:
    """
    Kite Connect API wrapper for InvestLLM.

    Supports both live trading and paper trading modes.
    """

    BASE_URL = "https://api.kite.trade"
    LOGIN_URL = "https://kite.zerodha.com/connect/login"

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        access_token: str = None,
        paper_trading: bool = True
    ):
        """
        Initialize Kite API client.

        Args:
            api_key: Kite Connect API key
            api_secret: Kite Connect API secret
            access_token: Access token (from login flow)
            paper_trading: If True, simulate orders without execution
        """
        self.api_key = api_key or os.getenv("ZERODHA_API_KEY", "")
        self.api_secret = api_secret or os.getenv("ZERODHA_API_SECRET", "")
        self.access_token = access_token or os.getenv("ZERODHA_ACCESS_TOKEN", "")
        self.paper_trading = paper_trading

        # Paper trading state
        self._paper_positions: Dict[str, Position] = {}
        self._paper_orders: List[Order] = []
        self._paper_capital = 100000.0
        self._paper_order_id = 1000

        # Session for API calls
        self.session = requests.Session()
        if self.access_token:
            self.session.headers.update({
                "Authorization": f"token {self.api_key}:{self.access_token}",
                "X-Kite-Version": "3"
            })

        mode = "PAPER" if paper_trading else "LIVE"
        logger.info(f"KiteAPI initialized in {mode} mode")

    # =========================================================================
    # AUTHENTICATION
    # =========================================================================

    def get_login_url(self) -> str:
        """Get the login URL for user authentication"""
        return f"{self.LOGIN_URL}?v=3&api_key={self.api_key}"

    def generate_session(self, request_token: str) -> str:
        """
        Generate access token from request token.

        Args:
            request_token: Token received after login redirect

        Returns:
            Access token for API calls
        """
        import hashlib

        checksum = hashlib.sha256(
            f"{self.api_key}{request_token}{self.api_secret}".encode()
        ).hexdigest()

        response = self.session.post(
            f"{self.BASE_URL}/session/token",
            data={
                "api_key": self.api_key,
                "request_token": request_token,
                "checksum": checksum
            }
        )

        if response.status_code == 200:
            data = response.json()
            self.access_token = data["data"]["access_token"]
            self.session.headers.update({
                "Authorization": f"token {self.api_key}:{self.access_token}"
            })
            logger.info("Session generated successfully")
            return self.access_token
        else:
            raise Exception(f"Session generation failed: {response.text}")

    # =========================================================================
    # MARKET DATA
    # =========================================================================

    def get_quote(self, symbol: str, exchange: str = "NSE") -> Quote:
        """
        Get real-time quote for a symbol.

        Args:
            symbol: Stock symbol (e.g., "RELIANCE")
            exchange: Exchange (NSE or BSE)

        Returns:
            Quote object with current market data
        """
        instrument = f"{exchange}:{symbol}"

        if self.paper_trading or not self.access_token:
            # Return simulated quote for paper trading
            return self._get_simulated_quote(symbol)

        response = self.session.get(
            f"{self.BASE_URL}/quote",
            params={"i": instrument}
        )

        if response.status_code == 200:
            data = response.json()["data"][instrument]
            return Quote(
                symbol=symbol,
                last_price=data["last_price"],
                open=data["ohlc"]["open"],
                high=data["ohlc"]["high"],
                low=data["ohlc"]["low"],
                close=data["ohlc"]["close"],
                volume=data["volume"],
                timestamp=datetime.now(),
                change=data.get("change", 0),
                change_percent=data.get("change_percent", 0)
            )
        else:
            raise Exception(f"Quote fetch failed: {response.text}")

    def get_quotes(self, symbols: List[str], exchange: str = "NSE") -> Dict[str, Quote]:
        """Get quotes for multiple symbols"""
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol, exchange)
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
        return quotes

    def _get_simulated_quote(self, symbol: str) -> Quote:
        """Generate simulated quote for paper trading"""
        import random

        # Use a deterministic base price based on symbol hash
        base_price = (hash(symbol) % 10000) / 10 + 100  # 100-1100 range

        # Add some randomness for daily movement
        daily_change = random.uniform(-0.02, 0.02)
        last_price = base_price * (1 + daily_change)

        return Quote(
            symbol=symbol,
            last_price=round(last_price, 2),
            open=round(base_price * 0.995, 2),
            high=round(base_price * 1.02, 2),
            low=round(base_price * 0.98, 2),
            close=round(base_price, 2),
            volume=random.randint(100000, 1000000),
            timestamp=datetime.now(),
            change=round(last_price - base_price, 2),
            change_percent=round(daily_change * 100, 2)
        )

    # =========================================================================
    # ORDER MANAGEMENT
    # =========================================================================

    def place_order(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        price: float = None,
        order_type: OrderType = OrderType.MARKET,
        product: ProductType = ProductType.CNC,
        exchange: str = "NSE"
    ) -> Order:
        """
        Place a buy or sell order.

        Args:
            symbol: Stock symbol
            transaction_type: BUY or SELL
            quantity: Number of shares
            price: Limit price (None for market orders)
            order_type: MARKET, LIMIT, SL, SL-M
            product: CNC (delivery), MIS (intraday), NRML (F&O)
            exchange: NSE or BSE

        Returns:
            Order object with order details
        """
        if self.paper_trading:
            return self._paper_place_order(
                symbol, transaction_type, quantity, price, order_type, product
            )

        # Live order placement
        order_params = {
            "tradingsymbol": symbol,
            "exchange": exchange,
            "transaction_type": transaction_type.value,
            "quantity": quantity,
            "order_type": order_type.value,
            "product": product.value,
            "validity": "DAY"
        }

        if price and order_type != OrderType.MARKET:
            order_params["price"] = price

        response = self.session.post(
            f"{self.BASE_URL}/orders/regular",
            data=order_params
        )

        if response.status_code == 200:
            order_id = response.json()["data"]["order_id"]
            logger.info(f"Order placed: {order_id} - {transaction_type.value} {quantity} {symbol}")

            return Order(
                order_id=order_id,
                symbol=symbol,
                transaction_type=transaction_type.value,
                quantity=quantity,
                price=price or 0,
                status="PENDING",
                order_type=order_type.value,
                timestamp=datetime.now()
            )
        else:
            raise Exception(f"Order placement failed: {response.text}")

    def _paper_place_order(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        price: float,
        order_type: OrderType,
        product: ProductType
    ) -> Order:
        """Simulate order placement for paper trading"""

        # Get current price
        quote = self._get_simulated_quote(symbol)
        exec_price = price if price else quote.last_price

        # Generate order ID
        order_id = f"PAPER_{self._paper_order_id}"
        self._paper_order_id += 1

        order = Order(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type.value,
            quantity=quantity,
            price=exec_price,
            status="COMPLETE",
            order_type=order_type.value,
            timestamp=datetime.now(),
            filled_quantity=quantity,
            average_price=exec_price
        )

        self._paper_orders.append(order)

        # Update paper positions
        self._update_paper_position(symbol, transaction_type, quantity, exec_price)

        logger.info(
            f"[PAPER] Order executed: {transaction_type.value} {quantity} {symbol} "
            f"@ {exec_price:.2f}"
        )

        return order

    def _update_paper_position(
        self,
        symbol: str,
        transaction_type: TransactionType,
        quantity: int,
        price: float
    ):
        """Update paper trading positions"""

        if symbol in self._paper_positions:
            pos = self._paper_positions[symbol]

            if transaction_type == TransactionType.BUY:
                # Add to position
                total_cost = (pos.average_price * pos.quantity) + (price * quantity)
                new_qty = pos.quantity + quantity
                pos.quantity = new_qty
                pos.average_price = total_cost / new_qty if new_qty > 0 else 0
                self._paper_capital -= price * quantity
            else:
                # Reduce position
                pos.quantity -= quantity
                self._paper_capital += price * quantity
                if pos.quantity <= 0:
                    del self._paper_positions[symbol]
                    return

            pos.last_price = price
            pos.pnl = (pos.last_price - pos.average_price) * pos.quantity
            pos.pnl_percent = ((pos.last_price / pos.average_price) - 1) * 100 if pos.average_price > 0 else 0

        elif transaction_type == TransactionType.BUY:
            # New position
            self._paper_positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                average_price=price,
                last_price=price,
                pnl=0,
                pnl_percent=0,
                product="CNC"
            )
            self._paper_capital -= price * quantity

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if self.paper_trading:
            logger.info(f"[PAPER] Order cancelled: {order_id}")
            return True

        response = self.session.delete(
            f"{self.BASE_URL}/orders/regular/{order_id}"
        )
        return response.status_code == 200

    def get_orders(self) -> List[Order]:
        """Get all orders for the day"""
        if self.paper_trading:
            return self._paper_orders

        response = self.session.get(f"{self.BASE_URL}/orders")
        if response.status_code == 200:
            orders = []
            for o in response.json()["data"]:
                orders.append(Order(
                    order_id=o["order_id"],
                    symbol=o["tradingsymbol"],
                    transaction_type=o["transaction_type"],
                    quantity=o["quantity"],
                    price=o.get("price", 0),
                    status=o["status"],
                    order_type=o["order_type"],
                    timestamp=datetime.fromisoformat(o["order_timestamp"]),
                    filled_quantity=o.get("filled_quantity", 0),
                    average_price=o.get("average_price", 0)
                ))
            return orders
        return []

    # =========================================================================
    # POSITIONS & HOLDINGS
    # =========================================================================

    def get_positions(self) -> List[Position]:
        """Get current positions"""
        if self.paper_trading:
            # Update PnL for paper positions
            for symbol, pos in self._paper_positions.items():
                quote = self._get_simulated_quote(symbol)
                pos.last_price = quote.last_price
                pos.pnl = (pos.last_price - pos.average_price) * pos.quantity
                pos.pnl_percent = ((pos.last_price / pos.average_price) - 1) * 100
            return list(self._paper_positions.values())

        response = self.session.get(f"{self.BASE_URL}/portfolio/positions")
        if response.status_code == 200:
            positions = []
            for p in response.json()["data"]["net"]:
                if p["quantity"] != 0:
                    positions.append(Position(
                        symbol=p["tradingsymbol"],
                        quantity=p["quantity"],
                        average_price=p["average_price"],
                        last_price=p["last_price"],
                        pnl=p["pnl"],
                        pnl_percent=(p["pnl"] / (p["average_price"] * abs(p["quantity"]))) * 100,
                        product=p["product"]
                    ))
            return positions
        return []

    def get_holdings(self) -> List[Position]:
        """Get long-term holdings (CNC)"""
        if self.paper_trading:
            return [p for p in self._paper_positions.values() if p.product == "CNC"]

        response = self.session.get(f"{self.BASE_URL}/portfolio/holdings")
        if response.status_code == 200:
            holdings = []
            for h in response.json()["data"]:
                holdings.append(Position(
                    symbol=h["tradingsymbol"],
                    quantity=h["quantity"],
                    average_price=h["average_price"],
                    last_price=h["last_price"],
                    pnl=h["pnl"],
                    pnl_percent=(h["pnl"] / (h["average_price"] * h["quantity"])) * 100,
                    product="CNC"
                ))
            return holdings
        return []

    # =========================================================================
    # ACCOUNT INFO
    # =========================================================================

    def get_margins(self) -> Dict:
        """Get account margins/funds"""
        if self.paper_trading:
            positions = self.get_positions()
            total_invested = sum(p.average_price * p.quantity for p in positions)
            total_pnl = sum(p.pnl for p in positions)

            return {
                "available_cash": self._paper_capital,
                "total_invested": total_invested,
                "total_pnl": total_pnl,
                "portfolio_value": self._paper_capital + total_invested + total_pnl
            }

        response = self.session.get(f"{self.BASE_URL}/user/margins")
        if response.status_code == 200:
            return response.json()["data"]["equity"]
        return {}

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def reset_paper_trading(self, initial_capital: float = 100000):
        """Reset paper trading state"""
        self._paper_positions = {}
        self._paper_orders = []
        self._paper_capital = initial_capital
        self._paper_order_id = 1000
        logger.info(f"Paper trading reset with capital: {initial_capital}")

    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary with all positions"""
        positions = self.get_positions()
        margins = self.get_margins()

        return {
            "mode": "PAPER" if self.paper_trading else "LIVE",
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.quantity,
                    "avg_price": p.average_price,
                    "ltp": p.last_price,
                    "pnl": p.pnl,
                    "pnl_pct": p.pnl_percent
                }
                for p in positions
            ],
            "total_positions": len(positions),
            "total_pnl": sum(p.pnl for p in positions),
            "available_cash": margins.get("available_cash", 0),
            "portfolio_value": margins.get("portfolio_value", 0)
        }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Paper trading example
    kite = KiteAPI(paper_trading=True)

    print("=" * 60)
    print("KITE API - Paper Trading Demo")
    print("=" * 60)

    # Get quote
    quote = kite.get_quote("RELIANCE")
    print(f"\nRELIANCE Quote: {quote.last_price}")

    # Place buy order
    order = kite.place_order(
        symbol="RELIANCE",
        transaction_type=TransactionType.BUY,
        quantity=10
    )
    print(f"\nBuy Order: {order.order_id} - {order.status}")

    # Check positions
    positions = kite.get_positions()
    print(f"\nPositions: {len(positions)}")
    for pos in positions:
        print(f"  {pos.symbol}: {pos.quantity} @ {pos.average_price:.2f}")

    # Get portfolio summary
    summary = kite.get_portfolio_summary()
    print(f"\nPortfolio Value: {summary['portfolio_value']:.2f}")
    print(f"Available Cash: {summary['available_cash']:.2f}")
