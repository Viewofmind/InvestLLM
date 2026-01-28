# Smart Exit Integration Guide

## Overview
This module provides advanced exit strategies to maximize returns and minimize drawdowns, solving the "Diamond Hands" vs "Stop Loss" dilemma.

## Components

### 1. `SmartExit` Strategy
Located in: `investllm/strategies/smart_exit.py`

**Features:**
- **Trailing Stop Loss:** Protects profits by moving the stop price up as the stock rises.
- **Moving Average Exit:** The "Smart" exit. Sells if price crosses below a key Moving Average (e.g., SMA 50), indicating a trend reversal.
- **Target Profit:** Simple fixed target exit.

### 2. `BaseBacktester`
Located in: `investllm/trading/backtester.py`

A modular engine that accepts any `SmartExit` configuration and runs it against historical data.

## Usage Example

```python
from investllm.strategies.smart_exit import SmartExit
from investllm.trading.backtester import BaseBacktester

# 1. Configure Strategy
# Example: No fixed Target, loose 25% Trailing Stop, but strict MA 50 Exit
strategy = SmartExit(
    trailing_stop_pct=0.25,      # 25% Stop Loss
    moving_avg_period=50,        # Exit if Price < SMA 50
    target_profit_pct=None       # Let winners run
)

# 2. Run Backtest
backtester = BaseBacktester()
results = backtester.run_backtest(
    ticker_data=my_data_dict,
    signals=my_signals_dict,
    exit_strategy=strategy
)

# 3. View Results
backtester.summary()
```

## Why this is better?
- **Standard Stop Loss (20%)**: Cuts volatile winners (like BEL) too early.
- **Smart Exit (SMA 50)**: Allows volatility but exits when the *Trend* actually breaks.
