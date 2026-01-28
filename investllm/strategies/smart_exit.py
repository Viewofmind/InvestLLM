import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

class SmartExit:
    """
    Implements advanced exit strategies:
    1. Trailing Stop Loss (Fixed %)
    2. Moving Average Crossover (Price < SMA_N)
    3. Target Profit (Fixed %)
    4. Time-based Exit (Hold for N days)
    """
    def __init__(
        self, 
        trailing_stop_pct: float = None,
        moving_avg_period: int = None,
        target_profit_pct: float = None,
        max_hold_days: int = None
    ):
        self.trailing_stop_pct = trailing_stop_pct
        self.moving_avg_period = moving_avg_period
        self.target_profit_pct = target_profit_pct
        self.max_hold_days = max_hold_days
        
    def check_exit(
        self, 
        entry_price: float, 
        current_price: float, 
        highest_price: float, 
        days_held: int,
        ma_series: pd.Series = None,
        current_idx: int = None
    ) -> Tuple[bool, str]:
        """
        Checks if any exit condition is met.
        Returns: (should_exit, reason)
        """
        
        # 1. Target Profit
        if self.target_profit_pct:
            pnl = (current_price - entry_price) / entry_price
            if pnl >= self.target_profit_pct:
                return True, f"Target Reached (+{pnl*100:.1f}%)"
                
        # 2. Maximum Holding Period
        if self.max_hold_days and days_held >= self.max_hold_days:
            return True, f"Max Hold ({days_held} days)"
            
        # 3. Trailing Stop Loss
        if self.trailing_stop_pct:
            # Trailing Stop Price = Highest Price * (1 - Stop %)
            stop_price = highest_price * (1 - self.trailing_stop_pct)
            if current_price < stop_price:
                dd = (current_price - highest_price) / highest_price
                return True, f"Trailing Stop ({dd*100:.1f}%)"
                
        # 4. Moving Average Crossover (Smart Exit)
        if self.moving_avg_period and ma_series is not None and current_idx is not None:
            # Check if Price crosses BELOW MA
            # Validation: Ensure we have enough data
            if current_idx >= self.moving_avg_period:
                current_ma = ma_series.iloc[current_idx]
                if current_price < current_ma:
                    return True, f"MA Cross (Price < SMA_{self.moving_avg_period})"
        
        return False, ""

    def simulate_trade(self, prices: pd.Series, entry_idx: int, ma_series: pd.Series = None) -> Dict:
        """
        Fast simulation of a single trade to find exit point.
        """
        entry_price = prices.iloc[entry_idx]
        highest = entry_price
        
        for i in range(entry_idx + 1, len(prices)):
            price = prices.iloc[i]
            highest = max(highest, price)
            days = i - entry_idx
            
            should_exit, reason = self.check_exit(
                entry_price, price, highest, days, ma_series, i
            )
            
            if should_exit:
                return {
                    "Exit Index": i,
                    "Exit Price": price,
                    "Reason": reason,
                    "PnL": (price - entry_price) / entry_price
                }
                
        # End of Data (Open Position)
        last_price = prices.iloc[-1]
        return {
            "Exit Index": len(prices) - 1,
            "Exit Price": last_price,
            "Reason": "Open (End of Data)",
            "PnL": (last_price - entry_price) / entry_price
        }
