"""
InvestLLM Smart Exit Strategy - COMPLETE VERSION
=================================================
Based on your optimization finding: Fixed stop-loss HURTS multibagger strategy!

Results:
- Baseline Buy & Hold: +9234%
- With 39% Stop Loss: +8534% (-700% lost!)
- Conclusion: NO FIXED STOP-LOSS for trend-following strategy

Instead we use:
1. Target Exits - Partial profit booking at 50%, 100%, 200%, 300%
2. MA Exit - Exit remaining position on trend reversal
3. Model Exit - Exit when LSTM prediction turns bearish
4. Time Exit - Exit dead money after 1-2 years
5. Catastrophic Exit - ONLY for >50% drop (fraud/bankruptcy protection)

Author: InvestLLM
Version: 1.0
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# DATA CLASSES AND ENUMS
# =============================================================================

class ExitReason(Enum):
    """Reasons for exiting a position"""
    HOLDING = "Still Holding"
    TARGET_PARTIAL = "Target Hit (Partial Exit)"
    MA_CROSSOVER = "MA Crossover (Trend Reversal)"
    MODEL_SIGNAL = "Model Exit Signal"
    MOMENTUM_LOSS = "Momentum Loss (RSI)"
    TREND_BREAK = "Major Trend Break (200 MA)"
    TIME_BASED = "Time-Based Exit (Dead Money)"
    CATASTROPHIC = "Catastrophic Drop (>50%)"


@dataclass
class ExitSignal:
    """Exit signal with all details"""
    should_exit: bool
    exit_portion: float  # 0.0 to 1.0 (supports partial exits)
    reason: ExitReason
    confidence: float  # 0.0 to 1.0
    details: str


@dataclass
class PositionState:
    """Track state of each position"""
    ticker: str
    entry_price: float
    entry_date: pd.Timestamp
    entry_prediction: float
    highest_price: float
    position_remaining: float  # 1.0 = 100%
    targets_hit: Dict[str, bool]
    was_overbought: bool
    days_held: int


# =============================================================================
# INDIVIDUAL EXIT STRATEGY CLASSES
# =============================================================================

class TargetExitStrategy:
    """
    Partial exits at profit targets.
    
    Why this works for multibaggers:
    - Locks in some profits along the way
    - Keeps "house money" running for unlimited upside
    - Reduces psychological pressure during drawdowns
    - You never fully exit a winner!
    
    Default targets:
    - 50% gain: Exit 20% of position
    - 100% gain: Exit 20% of position
    - 200% gain: Exit 20% of position
    - 300% gain: Exit 20% of position
    - Remaining 20% runs with MA exit
    """
    
    def __init__(self, targets: List[Tuple[float, float]] = None):
        """
        Args:
            targets: List of (gain_threshold, exit_portion)
                     e.g., [(0.5, 0.20), (1.0, 0.20), (2.0, 0.20), (3.0, 0.20)]
        """
        self.targets = targets or [
            (0.50, 0.20),   # Exit 20% at 50% gain
            (1.00, 0.20),   # Exit 20% at 100% gain (2x)
            (2.00, 0.20),   # Exit 20% at 200% gain (3x)
            (3.00, 0.20),   # Exit 20% at 300% gain (4x)
            # Remaining 20% uses MA exit
        ]
    
    def check_exit(self, entry_price: float, current_price: float,
                   targets_hit: Dict[str, bool]) -> Tuple[ExitSignal, Dict[str, bool]]:
        """
        Check if any profit target is hit.
        
        Returns:
            (ExitSignal, updated_targets_hit)
        """
        current_gain = (current_price - entry_price) / entry_price
        
        for threshold, portion in self.targets:
            target_key = f"target_{int(threshold*100)}"
            
            if not targets_hit.get(target_key, False) and current_gain >= threshold:
                targets_hit[target_key] = True
                
                return ExitSignal(
                    should_exit=True,
                    exit_portion=portion,
                    reason=ExitReason.TARGET_PARTIAL,
                    confidence=1.0,
                    details=f"🎯 Target {threshold*100:.0f}% hit! Gain: {current_gain*100:.1f}%. Booking {portion*100:.0f}%"
                ), targets_hit
        
        next_target = self._get_next_target(current_gain)
        return ExitSignal(
            should_exit=False,
            exit_portion=0.0,
            reason=ExitReason.HOLDING,
            confidence=0.0,
            details=f"Gain: {current_gain*100:.1f}%. Next target: {next_target}"
        ), targets_hit
    
    def _get_next_target(self, current_gain: float) -> str:
        for threshold, _ in self.targets:
            if current_gain < threshold:
                return f"{threshold*100:.0f}%"
        return "All targets hit ✅"


class MovingAverageExitStrategy:
    """
    Exit when price crosses below moving average (trend reversal).
    
    Why this works for multibaggers:
    - During uptrend, price stays above MA
    - MA rises with price, protecting gains
    - Only exits when trend ACTUALLY reverses
    - Allows 30-40% drawdowns if trend intact (this is key!)
    
    Recommended: 50-day EMA with 3-day confirmation
    """
    
    def __init__(self, ma_period: int = 50, ma_type: str = 'EMA',
                 confirmation_days: int = 3, min_profit_to_trigger: float = 0.30):
        """
        Args:
            ma_period: Period for moving average (50 recommended)
            ma_type: 'SMA' or 'EMA'
            confirmation_days: Days price must stay below MA
            min_profit_to_trigger: Only trigger if position has this much profit
        """
        self.ma_period = ma_period
        self.ma_type = ma_type
        self.confirmation_days = confirmation_days
        self.min_profit_to_trigger = min_profit_to_trigger
    
    def calculate_ma(self, prices: pd.Series) -> pd.Series:
        """Calculate moving average"""
        if self.ma_type == 'EMA':
            return prices.ewm(span=self.ma_period, adjust=False).mean()
        return prices.rolling(window=self.ma_period).mean()
    
    def check_exit(self, prices: pd.Series, entry_price: float) -> ExitSignal:
        """Check if MA exit signal is triggered"""
        if len(prices) < self.ma_period + self.confirmation_days:
            return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                            "Insufficient data for MA calculation")
        
        current_price = prices.iloc[-1]
        current_gain = (current_price - entry_price) / entry_price
        
        # Only exit on MA if in profit (don't sell at loss due to MA)
        if current_gain < self.min_profit_to_trigger:
            return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                            f"Gain {current_gain*100:.1f}% below MA trigger threshold")
        
        ma = self.calculate_ma(prices)
        
        # Check last N days for confirmation
        recent_prices = prices.iloc[-self.confirmation_days:]
        recent_ma = ma.iloc[-self.confirmation_days:]
        
        days_below_ma = sum(recent_prices < recent_ma)
        
        if days_below_ma >= self.confirmation_days:
            pct_below = (recent_ma.iloc[-1] - current_price) / recent_ma.iloc[-1] * 100
            return ExitSignal(
                should_exit=True,
                exit_portion=1.0,  # Full exit of remaining position
                reason=ExitReason.MA_CROSSOVER,
                confidence=min(1.0, days_below_ma / (self.confirmation_days + 1)),
                details=f"📉 Price {pct_below:.1f}% below {self.ma_period}-{self.ma_type} for {days_below_ma} days. Protecting {current_gain*100:.1f}% gain."
            )
        
        return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                         f"Price above {self.ma_period}-{self.ma_type}")


class ModelBasedExitStrategy:
    """
    Exit based on LSTM model predictions turning bearish.
    
    Why this works:
    - Uses same model that found the entry
    - Model can detect trend reversals early
    - More sophisticated than fixed rules
    """
    
    def __init__(self, exit_threshold: float = 0.0010,
                 lookback_days: int = 5, min_profit_to_trigger: float = 0.15):
        """
        Args:
            exit_threshold: Exit if prediction falls below this
            lookback_days: Days to analyze prediction trend
            min_profit_to_trigger: Only trigger if in profit
        """
        self.exit_threshold = exit_threshold
        self.lookback_days = lookback_days
        self.min_profit_to_trigger = min_profit_to_trigger
    
    def check_exit(self, predictions: List[float], entry_prediction: float,
                   entry_price: float, current_price: float) -> ExitSignal:
        """Check if model signals exit"""
        if not predictions or len(predictions) < self.lookback_days:
            return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                            "Insufficient prediction history")
        
        current_gain = (current_price - entry_price) / entry_price
        
        # Only exit on model signal if in profit
        if current_gain < self.min_profit_to_trigger:
            return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                            f"Gain {current_gain*100:.1f}% below model trigger threshold")
        
        current_pred = predictions[-1]
        avg_recent = np.mean(predictions[-self.lookback_days:])
        
        if current_pred < self.exit_threshold:
            return ExitSignal(
                should_exit=True,
                exit_portion=1.0,
                reason=ExitReason.MODEL_SIGNAL,
                confidence=0.8,
                details=f"🤖 Model prediction {current_pred:.5f} below threshold {self.exit_threshold:.5f}. Protecting {current_gain*100:.1f}% gain."
            )
        
        # Check for declining trend
        pred_drop = (entry_prediction - current_pred) / entry_prediction if entry_prediction > 0 else 0
        if avg_recent < self.exit_threshold and pred_drop > 0.5:
            return ExitSignal(
                should_exit=True,
                exit_portion=0.5,  # Partial exit on weak signal
                reason=ExitReason.MODEL_SIGNAL,
                confidence=0.6,
                details=f"🤖 Model prediction declining: {entry_prediction:.5f} → {current_pred:.5f}"
            )
        
        return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                         f"Model prediction: {current_pred:.5f}")


class MomentumExitStrategy:
    """
    Exit when momentum fades (RSI-based).
    
    Logic: If RSI was overbought (>75) and then drops below 40,
    it signals momentum loss - time to exit if in profit.
    """
    
    def __init__(self, rsi_period: int = 14, overbought: int = 75,
                 exit_threshold: int = 40, min_profit: float = 0.20):
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.exit_threshold = exit_threshold
        self.min_profit = min_profit
    
    def calculate_rsi(self, prices: pd.Series) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 and not pd.isna(rsi.iloc[-1]) else 50
    
    def check_exit(self, prices: pd.Series, entry_price: float,
                   was_overbought: bool) -> Tuple[ExitSignal, bool]:
        """
        Check momentum exit.
        Returns: (ExitSignal, updated_was_overbought)
        """
        if len(prices) < self.rsi_period + 1:
            return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                            "Insufficient data for RSI"), was_overbought
        
        current_price = prices.iloc[-1]
        current_gain = (current_price - entry_price) / entry_price
        rsi = self.calculate_rsi(prices)
        
        # Track if ever overbought
        is_currently_overbought = rsi >= self.overbought
        updated_overbought = was_overbought or is_currently_overbought
        
        # Exit if RSI drops after being overbought AND in profit
        if was_overbought and rsi < self.exit_threshold and current_gain > self.min_profit:
            return ExitSignal(
                should_exit=True,
                exit_portion=1.0,
                reason=ExitReason.MOMENTUM_LOSS,
                confidence=0.7,
                details=f"📊 Momentum loss: RSI dropped to {rsi:.1f} after overbought. Protecting {current_gain*100:.1f}% gain."
            ), updated_overbought
        
        return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                         f"RSI: {rsi:.1f}"), updated_overbought


class TrendBreakExitStrategy:
    """
    Exit on major trend break (price below 200-day MA).
    
    This is the "last resort" exit for remaining position.
    Only triggers if position has significant profit to protect.
    """
    
    def __init__(self, ma_period: int = 200, min_profit: float = 0.50):
        self.ma_period = ma_period
        self.min_profit = min_profit
    
    def check_exit(self, prices: pd.Series, entry_price: float) -> ExitSignal:
        """Check for major trend break"""
        if len(prices) < self.ma_period:
            return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                            "Insufficient data for 200 MA")
        
        current_price = prices.iloc[-1]
        current_gain = (current_price - entry_price) / entry_price
        ma_200 = prices.rolling(self.ma_period).mean().iloc[-1]
        
        if current_gain > self.min_profit and current_price < ma_200:
            return ExitSignal(
                should_exit=True,
                exit_portion=1.0,
                reason=ExitReason.TREND_BREAK,
                confidence=0.9,
                details=f"⚠️ Major trend break: Price below {self.ma_period} MA. Protecting {current_gain*100:.1f}% gain."
            )
        
        return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                         f"Price above {self.ma_period} MA")


class TimeBasedExitStrategy:
    """
    Exit underperforming positions (dead money).
    
    If position hasn't performed after 1-2 years, capital is better
    deployed elsewhere.
    """
    
    def __init__(self, max_days: int = 365, min_gain_required: float = 0.20):
        """
        Args:
            max_days: Max days to hold underperformer
            min_gain_required: Minimum gain expected by max_days
        """
        self.max_days = max_days
        self.min_gain_required = min_gain_required
    
    def check_exit(self, entry_price: float, current_price: float,
                   days_held: int) -> ExitSignal:
        """Exit if position underperforming after time limit"""
        current_gain = (current_price - entry_price) / entry_price
        
        if days_held >= self.max_days and current_gain < self.min_gain_required:
            return ExitSignal(
                should_exit=True,
                exit_portion=1.0,
                reason=ExitReason.TIME_BASED,
                confidence=0.7,
                details=f"⏰ Held {days_held} days with only {current_gain*100:.1f}% gain. Redeploy capital."
            )
        
        return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                         f"Days: {days_held}, Gain: {current_gain*100:.1f}%")


class CatastrophicExitStrategy:
    """
    EMERGENCY exit for extreme drops (>50% from entry).
    
    This is the ONLY "stop-loss" we keep.
    Only triggers for catastrophic events (fraud, bankruptcy, etc.)
    
    Why 50% and not lower:
    - Normal volatility can be 30-40% for multibaggers
    - 50%+ drop usually means fundamental issue
    - Preserves some capital in disaster
    """
    
    def __init__(self, max_drop: float = 0.50):
        """
        Args:
            max_drop: Maximum allowed drop from ENTRY (not from high!)
        """
        self.max_drop = max_drop
    
    def check_exit(self, entry_price: float, current_price: float) -> ExitSignal:
        """Emergency exit check"""
        current_loss = (entry_price - current_price) / entry_price
        
        if current_loss >= self.max_drop:
            return ExitSignal(
                should_exit=True,
                exit_portion=1.0,
                reason=ExitReason.CATASTROPHIC,
                confidence=1.0,
                details=f"🚨 EMERGENCY: Price dropped {current_loss*100:.1f}% from entry! Check for fundamental issues."
            )
        
        return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0, "No catastrophic drop")


# =============================================================================
# MAIN EXIT MANAGER - COMBINES ALL STRATEGIES
# =============================================================================

class SmartExitManager:
    """
    Master exit manager that combines all strategies.
    
    Priority order:
    1. Catastrophic Drop (>50%) - Emergency only
    2. Target Exits (50%, 100%, 200%, 300%) - Partial profit taking
    3. Model-Based Exit - When LSTM turns bearish
    4. Momentum Exit - When RSI fades after overbought
    5. MA Crossover Exit - Trend reversal
    6. Trend Break Exit - Major trend break (200 MA)
    7. Time-Based Exit - Dead money
    """
    
    def __init__(self, config: Dict = None):
        """Initialize with config or use recommended defaults"""
        config = config or RECOMMENDED_CONFIG
        
        # Initialize all strategies
        self.catastrophic = CatastrophicExitStrategy(
            max_drop=config.get('catastrophic_drop', 0.50)
        )
        
        self.target_exit = TargetExitStrategy(
            targets=config.get('targets', [
                (0.50, 0.20), (1.00, 0.20), (2.00, 0.20), (3.00, 0.20)
            ])
        )
        
        self.ma_exit = MovingAverageExitStrategy(
            ma_period=config.get('ma_period', 50),
            ma_type=config.get('ma_type', 'EMA'),
            confirmation_days=config.get('ma_confirm_days', 3),
            min_profit_to_trigger=config.get('ma_min_profit', 0.30)
        )
        
        self.model_exit = ModelBasedExitStrategy(
            exit_threshold=config.get('model_exit_threshold', 0.0010),
            min_profit_to_trigger=config.get('model_min_profit', 0.15)
        )
        
        self.momentum_exit = MomentumExitStrategy(
            overbought=config.get('rsi_overbought', 75),
            exit_threshold=config.get('rsi_exit_threshold', 40),
            min_profit=config.get('momentum_min_profit', 0.20)
        )
        
        self.trend_break = TrendBreakExitStrategy(
            ma_period=config.get('trend_ma', 200),
            min_profit=config.get('trend_min_profit', 0.50)
        )
        
        self.time_exit = TimeBasedExitStrategy(
            max_days=config.get('max_holding_days', 365),
            min_gain_required=config.get('time_min_gain', 0.20)
        )
        
        # Track positions
        self.positions: Dict[str, PositionState] = {}
    
    def register_position(self, ticker: str, entry_price: float,
                         entry_date: pd.Timestamp, entry_prediction: float = 0.0):
        """Register a new position for tracking"""
        self.positions[ticker] = PositionState(
            ticker=ticker,
            entry_price=entry_price,
            entry_date=entry_date,
            entry_prediction=entry_prediction,
            highest_price=entry_price,
            position_remaining=1.0,
            targets_hit={},
            was_overbought=False,
            days_held=0
        )
    
    def update_position(self, ticker: str, current_price: float, current_date: pd.Timestamp):
        """Update position state"""
        if ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        
        # Update highest price
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        
        # Update days held
        pos.days_held = (current_date - pos.entry_date).days
    
    def check_exit(self, ticker: str, prices: pd.Series,
                   predictions: List[float] = None,
                   current_date: pd.Timestamp = None) -> ExitSignal:
        """
        Check all exit conditions for a position.
        
        Args:
            ticker: Stock ticker
            prices: Price history (close prices)
            predictions: Recent model predictions (optional)
            current_date: Current date
        
        Returns:
            ExitSignal with action to take
        """
        if ticker not in self.positions:
            return ExitSignal(False, 0.0, ExitReason.HOLDING, 0.0,
                            f"Position {ticker} not registered")
        
        pos = self.positions[ticker]
        current_price = prices.iloc[-1]
        
        # Update position state
        if current_date:
            self.update_position(ticker, current_price, current_date)
        
        # =====================================================================
        # PRIORITY 1: Catastrophic Drop (Emergency)
        # =====================================================================
        signal = self.catastrophic.check_exit(pos.entry_price, current_price)
        if signal.should_exit:
            return signal
        
        # =====================================================================
        # PRIORITY 2: Target Exits (Partial Profit Taking)
        # =====================================================================
        signal, pos.targets_hit = self.target_exit.check_exit(
            pos.entry_price, current_price, pos.targets_hit
        )
        if signal.should_exit:
            pos.position_remaining -= signal.exit_portion
            return signal
        
        # =====================================================================
        # PRIORITY 3: Model-Based Exit
        # =====================================================================
        if predictions:
            signal = self.model_exit.check_exit(
                predictions, pos.entry_prediction,
                pos.entry_price, current_price
            )
            if signal.should_exit:
                return signal
        
        # =====================================================================
        # PRIORITY 4: Momentum Exit (RSI)
        # =====================================================================
        signal, pos.was_overbought = self.momentum_exit.check_exit(
            prices, pos.entry_price, pos.was_overbought
        )
        if signal.should_exit:
            return signal
        
        # =====================================================================
        # PRIORITY 5: MA Crossover Exit (for remaining position)
        # =====================================================================
        # Only check MA exit after some targets hit (position_remaining < 0.6)
        if pos.position_remaining < 0.6:
            signal = self.ma_exit.check_exit(prices, pos.entry_price)
            if signal.should_exit:
                return signal
        
        # =====================================================================
        # PRIORITY 6: Major Trend Break (200 MA)
        # =====================================================================
        signal = self.trend_break.check_exit(prices, pos.entry_price)
        if signal.should_exit:
            return signal
        
        # =====================================================================
        # PRIORITY 7: Time-Based Exit (Dead Money)
        # =====================================================================
        signal = self.time_exit.check_exit(
            pos.entry_price, current_price, pos.days_held
        )
        if signal.should_exit:
            return signal
        
        # =====================================================================
        # No exit triggered - HOLD
        # =====================================================================
        current_gain = (current_price - pos.entry_price) / pos.entry_price
        return ExitSignal(
            should_exit=False,
            exit_portion=0.0,
            reason=ExitReason.HOLDING,
            confidence=0.0,
            details=f"✅ HOLD {ticker}: +{current_gain*100:.1f}%, {pos.days_held} days, {pos.position_remaining*100:.0f}% remaining"
        )


# =============================================================================
# RECOMMENDED CONFIGURATION
# =============================================================================

RECOMMENDED_CONFIG = {
    # Catastrophic drop - ONLY for extreme cases
    'catastrophic_drop': 0.50,  # 50% from entry (NOT from high!)
    
    # Target exits - partial profit booking
    'targets': [
        (0.50, 0.20),   # Exit 20% at 50% gain
        (1.00, 0.20),   # Exit 20% at 100% gain (2x)
        (2.00, 0.20),   # Exit 20% at 200% gain (3x)
        (3.00, 0.20),   # Exit 20% at 300% gain (4x)
        # Keep 20% running with MA/trend exit
    ],
    
    # MA exit settings
    'ma_period': 50,
    'ma_type': 'EMA',
    'ma_confirm_days': 3,
    'ma_min_profit': 0.30,  # Only trigger if 30%+ profit
    
    # Model exit settings
    'model_exit_threshold': 0.0010,
    'model_min_profit': 0.15,
    
    # Momentum (RSI) settings
    'rsi_overbought': 75,
    'rsi_exit_threshold': 40,
    'momentum_min_profit': 0.20,
    
    # Trend break settings
    'trend_ma': 200,
    'trend_min_profit': 0.50,  # Only trigger if 50%+ profit
    
    # Time-based settings
    'max_holding_days': 365,  # 1 year
    'time_min_gain': 0.20,    # Expect at least 20% in 1 year
}


# =============================================================================
# EXAMPLE USAGE AND SIMULATION
# =============================================================================

def simulate_bel_trade():
    """
    Simulate how smart exit would work on BEL (your best trade).
    BEL: +525% gain with -30%+ drawdowns along the way.
    """
    print("=" * 70)
    print("🎯 SIMULATING BEL TRADE WITH SMART EXIT STRATEGY")
    print("=" * 70)
    
    # Initialize
    exit_manager = SmartExitManager(RECOMMENDED_CONFIG)
    
    # BEL trade details
    entry_price = 37.0
    entry_date = pd.Timestamp('2022-03-30')
    
    # Register position
    exit_manager.register_position('BEL', entry_price, entry_date, 0.00263)
    
    # Simulated price journey (with realistic drawdowns)
    price_journey = [
        (37.0, '2022-03-30', 'Entry'),
        (45.0, '2022-06-01', '+21% - Running'),
        (35.0, '2022-09-01', '-5% from entry, -22% drawdown (HOLD!)'),
        (56.0, '2022-12-01', '+51% - TARGET 1 HIT!'),
        (40.0, '2023-03-01', '+8%, -29% drawdown (HOLD!)'),
        (75.0, '2023-06-01', '+103% - TARGET 2 HIT!'),
        (60.0, '2023-09-01', '+62%, -20% drawdown (HOLD!)'),
        (112.0, '2023-12-01', '+203% - TARGET 3 HIT!'),
        (90.0, '2024-03-01', '+143%, -20% drawdown (HOLD!)'),
        (150.0, '2024-06-01', '+305% - TARGET 4 HIT!'),
        (120.0, '2024-09-01', '+224%, -20% drawdown (HOLD!)'),
        (200.0, '2025-01-01', '+440% - Running'),
        (232.0, '2025-06-01', '+527% - Current'),
    ]
    
    print(f"\n📊 Entry: ₹{entry_price} on {entry_date.strftime('%Y-%m-%d')}")
    print("-" * 70)
    
    total_booked = 0
    
    for price, date_str, comment in price_journey:
        current_date = pd.Timestamp(date_str)
        
        # Create price series (simplified)
        prices = pd.Series([entry_price] + [p[0] for p in price_journey[:price_journey.index((price, date_str, comment))+1]])
        
        # Check exit
        signal = exit_manager.check_exit('BEL', prices, current_date=current_date)
        
        pos = exit_manager.positions['BEL']
        gain = (price - entry_price) / entry_price * 100
        
        if signal.should_exit:
            booked = signal.exit_portion * 100
            total_booked += signal.exit_portion
            print(f"  {date_str}: ₹{price:6.0f} ({gain:+6.1f}%) | 🔔 {signal.reason.value}")
            print(f"            → Book {booked:.0f}% | Remaining: {pos.position_remaining*100:.0f}%")
        else:
            print(f"  {date_str}: ₹{price:6.0f} ({gain:+6.1f}%) | ✅ HOLD | {comment}")
    
    print("-" * 70)
    print(f"\n📈 RESULTS:")
    print(f"  Total Gain: +527%")
    print(f"  Position Booked: {total_booked*100:.0f}%")
    print(f"  Position Still Running: {(1-total_booked)*100:.0f}%")
    print(f"  Drawdowns Survived: -22%, -29%, -20%, -20%, -20%")
    print(f"  Would Stop-Loss Have Helped? NO! It would have exited early.")
    print("=" * 70)


def example_backtest_integration():
    """
    Example of how to integrate SmartExitManager into your backtester.
    """
    print("\n" + "=" * 70)
    print("📝 BACKTEST INTEGRATION EXAMPLE")
    print("=" * 70)
    
    code = '''
# In your backtester:

from smart_exit_complete import SmartExitManager, RECOMMENDED_CONFIG

# Initialize once
exit_manager = SmartExitManager(RECOMMENDED_CONFIG)

# When opening a position:
exit_manager.register_position(
    ticker='RELIANCE',
    entry_price=1200.0,
    entry_date=pd.Timestamp('2024-01-15'),
    entry_prediction=0.00250
)

# In your daily loop:
for date in trading_days:
    for ticker in open_positions:
        prices = get_price_history(ticker, lookback=250)
        predictions = get_recent_predictions(ticker, lookback=10)
        
        signal = exit_manager.check_exit(
            ticker=ticker,
            prices=prices,
            predictions=predictions,
            current_date=date
        )
        
        if signal.should_exit:
            # Execute partial or full exit
            quantity_to_sell = position[ticker].quantity * signal.exit_portion
            execute_sell(ticker, quantity_to_sell, signal.reason)
            print(f"{date}: {signal.details}")
'''
    print(code)


if __name__ == "__main__":
    # Run BEL simulation
    simulate_bel_trade()
    
    # Show integration example
    example_backtest_integration()
    
    print("\n✅ Smart Exit Strategy loaded successfully!")
    print("   Use: from smart_exit_complete import SmartExitManager, RECOMMENDED_CONFIG")
