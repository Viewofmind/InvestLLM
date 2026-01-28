"""
Risk Manager for InvestLLM
==========================
Portfolio-level risk management with real-time monitoring.

Features:
- Maximum Drawdown (MaxDD) monitoring and circuit breakers
- Position sizing based on risk budget
- Sector exposure limits
- Correlation-based risk adjustment
- Daily loss limits
- Volatility-based position scaling

Risk Rules:
- Max 5% capital per position
- Max 20% exposure per sector
- Daily loss limit: 3%
- Max drawdown circuit breaker: 15%
- Reduce position size by 50% if drawdown > 10%
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Risk alert types"""
    DRAWDOWN = "DRAWDOWN"
    DAILY_LOSS = "DAILY_LOSS"
    POSITION_SIZE = "POSITION_SIZE"
    SECTOR_EXPOSURE = "SECTOR_EXPOSURE"
    CONCENTRATION = "CONCENTRATION"
    VOLATILITY = "VOLATILITY"


@dataclass
class RiskAlert:
    """Risk alert details"""
    alert_type: AlertType
    level: RiskLevel
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    action_required: bool = False
    action: str = ""


@dataclass
class PositionRisk:
    """Risk metrics for a single position"""
    symbol: str
    value: float
    weight: float  # % of portfolio
    pnl: float
    pnl_pct: float
    sector: str
    volatility: float = 0.0
    var_95: float = 0.0  # Value at Risk (95%)
    max_loss: float = 0.0


@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    timestamp: datetime
    total_value: float
    cash: float
    invested: float

    # Drawdown metrics
    current_drawdown: float
    max_drawdown: float
    peak_value: float

    # Daily metrics
    daily_pnl: float
    daily_pnl_pct: float

    # Exposure metrics
    sector_exposure: Dict[str, float]
    concentration: float  # Herfindahl index
    top_position_weight: float

    # Risk level
    risk_level: RiskLevel
    alerts: List[RiskAlert] = field(default_factory=list)


class RiskManager:
    """
    Portfolio risk manager with real-time monitoring.

    Implements risk limits and circuit breakers to protect capital.
    """

    # Sector mapping for NSE stocks
    SECTOR_MAP = {
        "RELIANCE": "Energy",
        "ONGC": "Energy",
        "BPCL": "Energy",
        "COALINDIA": "Energy",
        "NTPC": "Power",
        "POWERGRID": "Power",
        "TATAPOWER": "Power",
        "TCS": "IT",
        "INFY": "IT",
        "WIPRO": "IT",
        "HCLTECH": "IT",
        "TECHM": "IT",
        "HDFCBANK": "Banking",
        "ICICIBANK": "Banking",
        "AXISBANK": "Banking",
        "SBIN": "Banking",
        "KOTAKBANK": "Banking",
        "INDUSINDBK": "Banking",
        "BAJFINANCE": "NBFC",
        "BAJAJFINSV": "NBFC",
        "HDFC": "NBFC",
        "MARUTI": "Auto",
        "TATAMOTORS": "Auto",
        "M&M": "Auto",
        "HEROMOTOCO": "Auto",
        "SUNPHARMA": "Pharma",
        "DRREDDY": "Pharma",
        "CIPLA": "Pharma",
        "DIVISLAB": "Pharma",
        "HINDUNILVR": "FMCG",
        "ITC": "FMCG",
        "NESTLEIND": "FMCG",
        "BRITANNIA": "FMCG",
        "TITAN": "Consumer",
        "ASIANPAINT": "Consumer",
        "TATASTEEL": "Metals",
        "JSWSTEEL": "Metals",
        "HINDALCO": "Metals",
        "ADANIENT": "Infra",
        "ADANIPORTS": "Infra",
        "LT": "Infra",
        "ULTRACEMCO": "Cement"
    }

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_pct: float = 0.05,      # 5% per position
        max_sector_pct: float = 0.20,         # 20% per sector
        max_drawdown_pct: float = 0.15,       # 15% circuit breaker
        daily_loss_limit_pct: float = 0.03,   # 3% daily loss limit
        drawdown_reduction_threshold: float = 0.10,  # Reduce at 10% DD
        position_reduction_factor: float = 0.50,     # Reduce by 50%
        data_dir: str = "data/risk"
    ):
        """
        Initialize risk manager.

        Args:
            initial_capital: Starting portfolio value
            max_position_pct: Maximum % per position
            max_sector_pct: Maximum % per sector
            max_drawdown_pct: Circuit breaker threshold
            daily_loss_limit_pct: Daily loss limit
            drawdown_reduction_threshold: Reduce positions at this drawdown
            position_reduction_factor: Position reduction multiplier
            data_dir: Directory for risk logs
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.drawdown_reduction_threshold = drawdown_reduction_threshold
        self.position_reduction_factor = position_reduction_factor

        # Data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # State tracking
        self.peak_value = initial_capital
        self.max_drawdown = 0.0
        self.daily_start_value = initial_capital
        self.last_reset_date = datetime.now().date()

        # Alert history
        self.alerts: List[RiskAlert] = []
        self.risk_history: List[PortfolioRisk] = []

        # Circuit breaker state
        self.circuit_breaker_triggered = False
        self.trading_halted = False

        logger.info(f"RiskManager initialized with capital: {initial_capital:,.0f}")

    # =========================================================================
    # RISK CALCULATIONS
    # =========================================================================

    def calculate_position_risk(
        self,
        symbol: str,
        quantity: int,
        avg_price: float,
        current_price: float,
        portfolio_value: float
    ) -> PositionRisk:
        """Calculate risk metrics for a single position"""
        value = quantity * current_price
        weight = value / portfolio_value if portfolio_value > 0 else 0
        pnl = (current_price - avg_price) * quantity
        pnl_pct = ((current_price / avg_price) - 1) * 100 if avg_price > 0 else 0
        sector = self.SECTOR_MAP.get(symbol, "Other")

        # Simple volatility estimate (would use historical data in production)
        volatility = 0.02  # 2% daily volatility assumption

        # 95% VaR (1.65 standard deviations)
        var_95 = value * volatility * 1.65

        # Maximum loss (using stop loss)
        max_loss = value * 0.15  # 15% stop loss

        return PositionRisk(
            symbol=symbol,
            value=value,
            weight=weight,
            pnl=pnl,
            pnl_pct=pnl_pct,
            sector=sector,
            volatility=volatility,
            var_95=var_95,
            max_loss=max_loss
        )

    def calculate_portfolio_risk(
        self,
        positions: List[Dict],
        cash: float
    ) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics.

        Args:
            positions: List of position dicts with symbol, quantity, avg_price, current_price
            cash: Available cash

        Returns:
            PortfolioRisk object with all risk metrics
        """
        # Calculate total values
        invested = sum(p['quantity'] * p['current_price'] for p in positions)
        total_value = invested + cash

        # Reset daily tracking if new day
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_start_value = total_value
            self.last_reset_date = today

        # Update peak and drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value

        current_drawdown = (self.peak_value - total_value) / self.peak_value if self.peak_value > 0 else 0

        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Daily P&L
        daily_pnl = total_value - self.daily_start_value
        daily_pnl_pct = daily_pnl / self.daily_start_value if self.daily_start_value > 0 else 0

        # Sector exposure
        sector_values = {}
        for p in positions:
            sector = self.SECTOR_MAP.get(p['symbol'], "Other")
            value = p['quantity'] * p['current_price']
            sector_values[sector] = sector_values.get(sector, 0) + value

        sector_exposure = {
            sector: value / total_value
            for sector, value in sector_values.items()
        } if total_value > 0 else {}

        # Concentration (Herfindahl-Hirschman Index)
        weights = [p['quantity'] * p['current_price'] / total_value for p in positions] if total_value > 0 else []
        concentration = sum(w ** 2 for w in weights) if weights else 0

        # Top position weight
        top_position_weight = max(weights) if weights else 0

        # Generate alerts
        alerts = self._generate_alerts(
            current_drawdown=current_drawdown,
            daily_pnl_pct=daily_pnl_pct,
            sector_exposure=sector_exposure,
            top_position_weight=top_position_weight,
            concentration=concentration
        )

        # Determine risk level
        risk_level = self._determine_risk_level(
            current_drawdown=current_drawdown,
            daily_pnl_pct=daily_pnl_pct,
            alerts=alerts
        )

        return PortfolioRisk(
            timestamp=datetime.now(),
            total_value=total_value,
            cash=cash,
            invested=invested,
            current_drawdown=current_drawdown,
            max_drawdown=self.max_drawdown,
            peak_value=self.peak_value,
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
            sector_exposure=sector_exposure,
            concentration=concentration,
            top_position_weight=top_position_weight,
            risk_level=risk_level,
            alerts=alerts
        )

    def _generate_alerts(
        self,
        current_drawdown: float,
        daily_pnl_pct: float,
        sector_exposure: Dict[str, float],
        top_position_weight: float,
        concentration: float
    ) -> List[RiskAlert]:
        """Generate risk alerts based on current metrics"""
        alerts = []

        # Drawdown alerts
        if current_drawdown >= self.max_drawdown_pct:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                level=RiskLevel.CRITICAL,
                message=f"CIRCUIT BREAKER: Drawdown {current_drawdown:.1%} exceeds limit {self.max_drawdown_pct:.1%}",
                value=current_drawdown,
                threshold=self.max_drawdown_pct,
                action_required=True,
                action="HALT_TRADING"
            ))
            self.circuit_breaker_triggered = True
            self.trading_halted = True

        elif current_drawdown >= self.drawdown_reduction_threshold:
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                level=RiskLevel.HIGH,
                message=f"Drawdown {current_drawdown:.1%} - reduce position sizes by {(1-self.position_reduction_factor):.0%}",
                value=current_drawdown,
                threshold=self.drawdown_reduction_threshold,
                action_required=True,
                action="REDUCE_POSITIONS"
            ))

        elif current_drawdown >= 0.05:  # 5% warning
            alerts.append(RiskAlert(
                alert_type=AlertType.DRAWDOWN,
                level=RiskLevel.MEDIUM,
                message=f"Drawdown warning: {current_drawdown:.1%}",
                value=current_drawdown,
                threshold=0.05
            ))

        # Daily loss alert
        if daily_pnl_pct <= -self.daily_loss_limit_pct:
            alerts.append(RiskAlert(
                alert_type=AlertType.DAILY_LOSS,
                level=RiskLevel.CRITICAL,
                message=f"Daily loss limit breached: {daily_pnl_pct:.1%}",
                value=daily_pnl_pct,
                threshold=-self.daily_loss_limit_pct,
                action_required=True,
                action="HALT_NEW_TRADES"
            ))

        # Sector exposure alerts
        for sector, exposure in sector_exposure.items():
            if exposure > self.max_sector_pct:
                alerts.append(RiskAlert(
                    alert_type=AlertType.SECTOR_EXPOSURE,
                    level=RiskLevel.HIGH,
                    message=f"{sector} exposure {exposure:.1%} exceeds limit {self.max_sector_pct:.1%}",
                    value=exposure,
                    threshold=self.max_sector_pct,
                    action_required=True,
                    action=f"REDUCE_{sector.upper()}"
                ))

        # Position concentration alert
        if top_position_weight > self.max_position_pct * 2:  # 2x limit
            alerts.append(RiskAlert(
                alert_type=AlertType.CONCENTRATION,
                level=RiskLevel.HIGH,
                message=f"Top position weight {top_position_weight:.1%} exceeds 2x limit",
                value=top_position_weight,
                threshold=self.max_position_pct * 2,
                action_required=True,
                action="REBALANCE"
            ))

        return alerts

    def _determine_risk_level(
        self,
        current_drawdown: float,
        daily_pnl_pct: float,
        alerts: List[RiskAlert]
    ) -> RiskLevel:
        """Determine overall portfolio risk level"""
        critical_alerts = [a for a in alerts if a.level == RiskLevel.CRITICAL]
        high_alerts = [a for a in alerts if a.level == RiskLevel.HIGH]

        if critical_alerts or current_drawdown >= self.max_drawdown_pct:
            return RiskLevel.CRITICAL
        elif high_alerts or current_drawdown >= self.drawdown_reduction_threshold:
            return RiskLevel.HIGH
        elif current_drawdown >= 0.05 or daily_pnl_pct <= -0.02:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # =========================================================================
    # POSITION SIZING
    # =========================================================================

    def calculate_position_size(
        self,
        symbol: str,
        current_price: float,
        signal_strength: float,
        portfolio_value: float,
        existing_positions: List[Dict]
    ) -> Tuple[int, str]:
        """
        Calculate optimal position size with risk constraints.

        Args:
            symbol: Stock symbol
            current_price: Current stock price
            signal_strength: Model signal strength (0-1)
            portfolio_value: Total portfolio value
            existing_positions: List of existing position dicts

        Returns:
            (quantity, reason) - 0 if position not allowed
        """
        # Check circuit breaker
        if self.trading_halted:
            return 0, "Trading halted - circuit breaker triggered"

        # Base position size
        base_value = portfolio_value * self.max_position_pct

        # Adjust by signal strength
        adjusted_value = base_value * min(signal_strength, 1.0)

        # Apply drawdown reduction if needed
        current_dd = (self.peak_value - portfolio_value) / self.peak_value if self.peak_value > 0 else 0

        if current_dd >= self.drawdown_reduction_threshold:
            adjusted_value *= self.position_reduction_factor
            logger.info(f"Position reduced due to drawdown: {current_dd:.1%}")

        # Check sector exposure
        sector = self.SECTOR_MAP.get(symbol, "Other")
        sector_value = sum(
            p['quantity'] * p['current_price']
            for p in existing_positions
            if self.SECTOR_MAP.get(p['symbol'], "Other") == sector
        )
        sector_exposure = sector_value / portfolio_value if portfolio_value > 0 else 0

        if sector_exposure + (adjusted_value / portfolio_value) > self.max_sector_pct:
            # Limit to stay within sector cap
            max_additional = (self.max_sector_pct - sector_exposure) * portfolio_value
            if max_additional <= 0:
                return 0, f"Sector exposure limit reached for {sector}"
            adjusted_value = min(adjusted_value, max_additional)

        # Calculate quantity
        quantity = int(adjusted_value / current_price)

        if quantity < 1:
            return 0, "Position size too small"

        return quantity, "OK"

    # =========================================================================
    # MONITORING
    # =========================================================================

    def monitor(
        self,
        positions: List[Dict],
        cash: float
    ) -> PortfolioRisk:
        """
        Run risk monitoring and return current state.

        Args:
            positions: Current positions
            cash: Available cash

        Returns:
            PortfolioRisk with current state and alerts
        """
        risk = self.calculate_portfolio_risk(positions, cash)

        # Store in history
        self.risk_history.append(risk)

        # Store alerts
        self.alerts.extend(risk.alerts)

        # Log if high risk
        if risk.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            logger.warning(f"RISK LEVEL: {risk.risk_level.value}")
            for alert in risk.alerts:
                logger.warning(f"  [{alert.level.value}] {alert.message}")

        return risk

    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual action)"""
        self.circuit_breaker_triggered = False
        self.trading_halted = False
        logger.info("Circuit breaker reset - trading enabled")

    def reset_daily(self, current_value: float):
        """Reset daily tracking"""
        self.daily_start_value = current_value
        self.last_reset_date = datetime.now().date()
        logger.info(f"Daily reset: starting value {current_value:,.0f}")

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_summary(self) -> Dict:
        """Get current risk summary"""
        latest_risk = self.risk_history[-1] if self.risk_history else None

        return {
            "risk_level": latest_risk.risk_level.value if latest_risk else "UNKNOWN",
            "current_drawdown": f"{(latest_risk.current_drawdown * 100) if latest_risk else 0:.1f}%",
            "max_drawdown": f"{self.max_drawdown * 100:.1f}%",
            "daily_pnl": f"{(latest_risk.daily_pnl_pct * 100) if latest_risk else 0:.1f}%",
            "peak_value": self.peak_value,
            "circuit_breaker": self.circuit_breaker_triggered,
            "trading_halted": self.trading_halted,
            "active_alerts": len([a for a in self.alerts if a.action_required]),
            "limits": {
                "max_position": f"{self.max_position_pct:.0%}",
                "max_sector": f"{self.max_sector_pct:.0%}",
                "max_drawdown": f"{self.max_drawdown_pct:.0%}",
                "daily_loss": f"{self.daily_loss_limit_pct:.0%}"
            }
        }

    def get_alerts(self, active_only: bool = True) -> List[Dict]:
        """Get risk alerts"""
        alerts = self.alerts
        if active_only:
            alerts = [a for a in alerts if a.action_required]
        return [asdict(a) for a in alerts]

    def save_report(self, filename: str = None):
        """Save risk report to file"""
        if filename is None:
            filename = f"risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = self.data_dir / filename

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_summary(),
            "alerts": self.get_alerts(active_only=False),
            "history": [
                {
                    "timestamp": r.timestamp.isoformat(),
                    "total_value": r.total_value,
                    "current_drawdown": r.current_drawdown,
                    "daily_pnl_pct": r.daily_pnl_pct,
                    "risk_level": r.risk_level.value
                }
                for r in self.risk_history[-100:]
            ]
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Risk report saved to {filepath}")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Initialize risk manager
    rm = RiskManager(
        initial_capital=100000,
        max_position_pct=0.05,
        max_sector_pct=0.20,
        max_drawdown_pct=0.15
    )

    print("=" * 60)
    print("RISK MANAGER DEMO")
    print("=" * 60)

    # Simulate positions
    positions = [
        {"symbol": "RELIANCE", "quantity": 50, "avg_price": 2500, "current_price": 2450},
        {"symbol": "TCS", "quantity": 20, "avg_price": 3500, "current_price": 3600},
        {"symbol": "HDFCBANK", "quantity": 30, "avg_price": 1600, "current_price": 1550},
        {"symbol": "INFY", "quantity": 40, "avg_price": 1400, "current_price": 1380},
    ]
    cash = 20000

    # Run monitoring
    risk = rm.monitor(positions, cash)

    print(f"\nPortfolio Value: {risk.total_value:,.0f}")
    print(f"Current Drawdown: {risk.current_drawdown:.1%}")
    print(f"Max Drawdown: {risk.max_drawdown:.1%}")
    print(f"Daily P&L: {risk.daily_pnl:+,.0f} ({risk.daily_pnl_pct:.1%})")
    print(f"Risk Level: {risk.risk_level.value}")

    print("\nSector Exposure:")
    for sector, exposure in sorted(risk.sector_exposure.items(), key=lambda x: -x[1]):
        print(f"  {sector}: {exposure:.1%}")

    if risk.alerts:
        print("\nAlerts:")
        for alert in risk.alerts:
            print(f"  [{alert.level.value}] {alert.message}")

    # Test position sizing
    print("\n" + "-" * 60)
    print("Position Sizing Test")
    print("-" * 60)

    qty, reason = rm.calculate_position_size(
        symbol="WIPRO",
        current_price=400,
        signal_strength=0.8,
        portfolio_value=risk.total_value,
        existing_positions=positions
    )
    print(f"WIPRO position size: {qty} shares ({reason})")

    # Summary
    print("\n" + "-" * 60)
    print("Summary")
    print("-" * 60)
    for key, value in rm.get_summary().items():
        print(f"  {key}: {value}")

    # Save report
    rm.save_report()
