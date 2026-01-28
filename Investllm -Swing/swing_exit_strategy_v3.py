"""
Swing Trading Exit Strategy V3 - FULLY OPTIMIZED
Based on V2 backtest analysis showing STOP_LOSS_4%_LOW causing major losses

Key Changes from V2:
1. Loosened low confidence stop (4% -> 5%) to reduce whipsaws
2. Volatility-adjusted position sizing
3. ATR-based dynamic profit targets
4. Partial profit taking (scale out)
5. Market regime awareness
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class TradeConfigV3:
    """Configuration for swing trade exits - FULLY OPTIMIZED"""

    # Profit targets based on confidence
    profit_high_confidence: float = 0.25    # >80% confidence -> 25%
    profit_medium_confidence: float = 0.20  # 60-80% confidence -> 20%
    profit_low_confidence: float = 0.15     # <60% confidence -> 15%

    # Stop losses - LOOSENED for low confidence (was 4%, now 5%)
    stop_loss_high_confidence: float = 0.06  # High confidence = more room
    stop_loss_medium_confidence: float = 0.05  # Medium = standard
    stop_loss_low_confidence: float = 0.05  # Low confidence = was 4%, now 5%

    # Volatility adjustments
    volatility_stop_multiplier: float = 1.3  # Reduced from 1.5
    high_volatility_threshold: float = 0.35  # ATR/price > 35%

    # Time-based
    max_hold_days: int = 7

    # Trailing stop - IMPROVED
    trailing_stop_activation: float = 0.08  # Activate at 8%
    trailing_stop_distance: float = 0.04    # Trail by 4%

    # Partial profit taking (NEW)
    partial_take_profit_1: float = 0.08  # Take 33% at 8%
    partial_take_profit_2: float = 0.12  # Take 33% at 12%
    partial_exit_pct: float = 0.33  # Exit 33% at each level

    # Breakeven protection
    breakeven_activation: float = 0.05  # Move to breakeven after 5%

    # Position sizing by volatility (NEW)
    position_size_high_vol: float = 0.50   # 50% for high vol stocks
    position_size_medium_vol: float = 0.75  # 75% for medium vol
    position_size_low_vol: float = 1.00    # 100% for low vol

    # Stock filter - exclude chronic losers
    excluded_stocks: list = field(default_factory=lambda: [
        'ATGL', 'OFSS', 'ICICIGI', 'BERGEPAINT'
    ])


class SwingExitStrategyV3:
    """
    FULLY OPTIMIZED exit strategy V3

    Key improvements over V2:
    1. Loosened low confidence stop (5% vs 4%)
    2. Volatility-based position sizing
    3. Partial profit taking
    4. ATR-based dynamic targets
    """

    def __init__(self, config: TradeConfigV3 = None):
        self.config = config or TradeConfigV3()
        self.active_trades = {}

    def determine_profit_target(self, confidence: float, atr_pct: float = None) -> float:
        """
        Determine profit target based on confidence AND volatility
        High volatility stocks can move more, so larger targets
        """
        # Base target by confidence
        if confidence >= 0.80:
            base_target = self.config.profit_high_confidence
        elif confidence >= 0.60:
            base_target = self.config.profit_medium_confidence
        else:
            base_target = self.config.profit_low_confidence

        # ATR-based adjustment (if provided)
        if atr_pct is not None and atr_pct > 0.02:
            # For volatile stocks, use ATR * 3 as minimum target
            atr_target = atr_pct * 3
            return max(base_target, atr_target)

        return base_target

    def determine_stop_loss(self, confidence: float, volatility: float) -> float:
        """
        V3: Loosened low confidence stop from 4% to 5%
        """
        # Base stop by confidence
        if confidence >= 0.80:
            base_stop = self.config.stop_loss_high_confidence
        elif confidence >= 0.60:
            base_stop = self.config.stop_loss_medium_confidence
        else:
            base_stop = self.config.stop_loss_low_confidence  # Now 5%

        # Volatility adjustment (high vol = wider stop)
        if volatility > self.config.high_volatility_threshold:
            return base_stop * self.config.volatility_stop_multiplier

        return base_stop

    def calculate_position_size_multiplier(
        self,
        confidence: float,
        volatility: float
    ) -> float:
        """
        NEW: Calculate position size based on BOTH confidence and volatility

        Logic:
        - High confidence + Low volatility = Full size (100%)
        - Low confidence + High volatility = Small size (25%)
        """
        # Confidence multiplier
        if confidence >= 0.80:
            conf_mult = 1.0
        elif confidence >= 0.60:
            conf_mult = 0.7
        else:
            conf_mult = 0.5

        # Volatility multiplier
        if volatility > 0.40:  # High vol
            vol_mult = self.config.position_size_high_vol
        elif volatility > 0.25:  # Medium vol
            vol_mult = self.config.position_size_medium_vol
        else:  # Low vol
            vol_mult = self.config.position_size_low_vol

        # Combined multiplier (multiply both factors)
        return conf_mult * vol_mult

    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float
    ) -> Optional[float]:
        """Calculate trailing stop price"""
        profit = (peak_price - entry_price) / entry_price

        if profit < self.config.trailing_stop_activation:
            return None

        trailing_stop = peak_price * (1 - self.config.trailing_stop_distance)
        return trailing_stop

    def check_partial_exit(
        self,
        entry_price: float,
        current_price: float,
        partial_exits_done: int
    ) -> Tuple[bool, float, int]:
        """
        NEW: Check if we should take partial profits

        Returns: (should_exit_partial, exit_pct, new_partial_exits_done)
        """
        profit_pct = (current_price - entry_price) / entry_price

        # First partial at 8%
        if partial_exits_done == 0 and profit_pct >= self.config.partial_take_profit_1:
            return True, self.config.partial_exit_pct, 1

        # Second partial at 12%
        if partial_exits_done == 1 and profit_pct >= self.config.partial_take_profit_2:
            return True, self.config.partial_exit_pct, 2

        return False, 0.0, partial_exits_done

    def should_exit(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float,
        entry_date: datetime,
        current_date: datetime,
        confidence: float,
        volatility: float,
        atr_pct: float = None
    ) -> Tuple[bool, str, float]:
        """
        V3: Fully optimized exit logic
        """
        # Calculate current profit/loss
        profit_pct = (current_price - entry_price) / entry_price

        # Handle timezone-aware dates properly
        try:
            days_held = (pd.Timestamp(current_date) - pd.Timestamp(entry_date)).days
        except:
            days_held = (current_date - entry_date).days

        # 1. Check profit target (now ATR-aware)
        profit_target = self.determine_profit_target(confidence, atr_pct)
        if profit_pct >= profit_target:
            return True, f"PROFIT_TARGET_{profit_target*100:.0f}%", current_price

        # 2. Check stop loss (V3: loosened for low confidence)
        stop_loss = self.determine_stop_loss(confidence, volatility)
        if profit_pct <= -stop_loss:
            conf_level = "HIGH" if confidence >= 0.80 else "MED" if confidence >= 0.60 else "LOW"
            return True, f"STOP_LOSS_{stop_loss*100:.0f}%_{conf_level}", current_price

        # 3. Check trailing stop
        trailing_stop = self.calculate_trailing_stop(entry_price, current_price, peak_price)
        if trailing_stop and current_price <= trailing_stop:
            locked_profit = (trailing_stop - entry_price) / entry_price
            return True, f"TRAILING_STOP_LOCKED_{locked_profit*100:.1f}%", current_price

        # 4. Check breakeven protection
        peak_profit = (peak_price - entry_price) / entry_price
        if peak_profit >= self.config.breakeven_activation:
            if profit_pct <= 0.005:
                return True, "BREAKEVEN_PROTECTION", current_price

        # 5. Max hold
        if days_held >= self.config.max_hold_days:
            return True, f"MAX_HOLD_{self.config.max_hold_days}D", current_price

        # 6. No exit - HOLD
        return False, "HOLD", current_price

    def is_stock_allowed(self, symbol: str) -> bool:
        """Check if stock is allowed (not in exclusion list)"""
        return symbol not in self.config.excluded_stocks


class TradeMonitorV3:
    """Enhanced trade monitor with partial exits and better tracking"""

    def __init__(self, strategy: SwingExitStrategyV3):
        self.strategy = strategy
        self.active_trades = {}
        self.trade_history = []
        self.partial_exits = []  # Track partial exits

    def add_trade(
        self,
        trade_id: str,
        symbol: str,
        entry_price: float,
        entry_date: datetime,
        quantity: int,
        confidence: float,
        volatility: float,
        atr_pct: float = None
    ):
        """Add a new trade with V3 optimizations"""

        # Calculate targets based on V3 strategy
        profit_target = self.strategy.determine_profit_target(confidence, atr_pct)
        stop_loss = self.strategy.determine_stop_loss(confidence, volatility)
        position_multiplier = self.strategy.calculate_position_size_multiplier(
            confidence, volatility
        )

        # Adjust quantity by position multiplier
        adjusted_quantity = int(quantity * position_multiplier)

        self.active_trades[trade_id] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'original_quantity': quantity,
            'current_quantity': adjusted_quantity,
            'confidence': confidence,
            'volatility': volatility,
            'atr_pct': atr_pct,
            'peak_price': entry_price,
            'lowest_price': entry_price,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'position_multiplier': position_multiplier,
            'status': 'ACTIVE',
            'partial_exits_done': 0,
            'partial_profit_locked': 0.0,
            'breakeven_triggered': False,
            'trailing_triggered': False
        }

        self.trade_history.append({
            'trade_id': trade_id,
            'action': 'ENTRY',
            'date': entry_date,
            'price': entry_price,
            'quantity': adjusted_quantity,
            'details': f"Conf={confidence:.2f}, Vol={volatility:.2f}, Size={position_multiplier:.0%}"
        })

    def update_trade(
        self,
        trade_id: str,
        current_price: float,
        current_date: datetime
    ) -> Optional[Dict]:
        """Update trade with partial exit support"""

        if trade_id not in self.active_trades:
            return None

        trade = self.active_trades[trade_id]

        # Update price tracking
        if current_price > trade['peak_price']:
            trade['peak_price'] = current_price
        if current_price < trade['lowest_price']:
            trade['lowest_price'] = current_price

        # Track triggers
        peak_profit = (trade['peak_price'] - trade['entry_price']) / trade['entry_price']
        if peak_profit >= self.strategy.config.breakeven_activation:
            trade['breakeven_triggered'] = True
        if peak_profit >= self.strategy.config.trailing_stop_activation:
            trade['trailing_triggered'] = True

        # Check for partial exits first
        should_partial, exit_pct, new_partial_count = self.strategy.check_partial_exit(
            trade['entry_price'],
            current_price,
            trade['partial_exits_done']
        )

        if should_partial and trade['current_quantity'] > 0:
            partial_quantity = int(trade['current_quantity'] * exit_pct)
            if partial_quantity > 0:
                profit_pct = (current_price - trade['entry_price']) / trade['entry_price']

                # Record partial exit
                self.partial_exits.append({
                    'trade_id': trade_id,
                    'date': current_date,
                    'price': current_price,
                    'quantity': partial_quantity,
                    'profit_pct': profit_pct
                })

                # Update trade
                trade['current_quantity'] -= partial_quantity
                trade['partial_exits_done'] = new_partial_count
                trade['partial_profit_locked'] += profit_pct * partial_quantity * trade['entry_price']

                self.trade_history.append({
                    'trade_id': trade_id,
                    'action': f'PARTIAL_EXIT_{new_partial_count}',
                    'date': current_date,
                    'price': current_price,
                    'quantity': partial_quantity,
                    'details': f"Locked {profit_pct*100:+.1f}% on {partial_quantity} shares"
                })

        # Check full exit conditions
        should_exit, reason, exit_price = self.strategy.should_exit(
            entry_price=trade['entry_price'],
            current_price=current_price,
            peak_price=trade['peak_price'],
            entry_date=trade['entry_date'],
            current_date=current_date,
            confidence=trade['confidence'],
            volatility=trade['volatility'],
            atr_pct=trade.get('atr_pct')
        )

        if should_exit and trade['current_quantity'] > 0:
            # Calculate final P&L including partial exits
            final_profit_pct = (exit_price - trade['entry_price']) / trade['entry_price']
            final_profit_amount = final_profit_pct * trade['entry_price'] * trade['current_quantity']
            total_profit = final_profit_amount + trade['partial_profit_locked']

            exit_signal = {
                'trade_id': trade_id,
                'symbol': trade['symbol'],
                'entry_price': trade['entry_price'],
                'exit_price': exit_price,
                'entry_date': trade['entry_date'],
                'exit_date': current_date,
                'original_quantity': trade['original_quantity'],
                'final_quantity': trade['current_quantity'],
                'profit_pct': final_profit_pct,
                'total_profit_amount': total_profit,
                'exit_reason': reason,
                'days_held': (current_date - trade['entry_date']).days,
                'peak_price': trade['peak_price'],
                'confidence': trade['confidence'],
                'volatility': trade['volatility'],
                'position_multiplier': trade['position_multiplier'],
                'partial_exits_done': trade['partial_exits_done'],
                'partial_profit_locked': trade['partial_profit_locked']
            }

            trade['status'] = 'CLOSED'

            self.trade_history.append({
                'trade_id': trade_id,
                'action': 'EXIT',
                'date': current_date,
                'price': exit_price,
                'quantity': trade['current_quantity'],
                'details': f"{reason}, Total P&L=₹{total_profit:+,.0f}"
            })

            return exit_signal

        return None

    def get_active_trades(self) -> Dict:
        return {k: v for k, v in self.active_trades.items() if v['status'] == 'ACTIVE'}

    def get_trade_summary(self) -> Dict:
        """Get summary statistics"""
        closed = [t for t in self.active_trades.values() if t['status'] == 'CLOSED']
        return {
            'total_trades': len(self.active_trades),
            'active_trades': len(self.get_active_trades()),
            'closed_trades': len(closed),
            'partial_exits': len(self.partial_exits),
            'history_entries': len(self.trade_history)
        }


def main():
    """Test V3 exit strategy"""
    print("\n" + "="*60)
    print("Swing Exit Strategy V3 - FULLY OPTIMIZED")
    print("="*60)

    config = TradeConfigV3()
    strategy = SwingExitStrategyV3(config)

    print("\nKey V3 Changes from V2:")
    print("  1. Loosened low conf stop: 4% -> 5% (reduce whipsaws)")
    print("  2. Volatility-based position sizing")
    print("  3. Partial profit taking at 8% and 12%")
    print("  4. ATR-based dynamic profit targets")

    print("\nStop Loss Configuration:")
    print(f"  - High confidence: {config.stop_loss_high_confidence*100:.0f}%")
    print(f"  - Medium confidence: {config.stop_loss_medium_confidence*100:.0f}%")
    print(f"  - Low confidence: {config.stop_loss_low_confidence*100:.0f}% (was 4% in V2)")

    print("\nPartial Profit Taking:")
    print(f"  - At +{config.partial_take_profit_1*100:.0f}%: Exit {config.partial_exit_pct*100:.0f}%")
    print(f"  - At +{config.partial_take_profit_2*100:.0f}%: Exit {config.partial_exit_pct*100:.0f}%")
    print(f"  - At target: Exit remaining")

    # Demo position sizing
    print("\n" + "-"*60)
    print("Position Sizing Examples:")
    print("-"*60)

    test_cases = [
        (0.85, 0.20, "High conf + Low vol"),
        (0.85, 0.45, "High conf + High vol"),
        (0.45, 0.20, "Low conf + Low vol"),
        (0.45, 0.45, "Low conf + High vol"),
    ]

    for conf, vol, desc in test_cases:
        mult = strategy.calculate_position_size_multiplier(conf, vol)
        print(f"  {desc}: {mult*100:.0f}% position")


if __name__ == '__main__':
    main()
