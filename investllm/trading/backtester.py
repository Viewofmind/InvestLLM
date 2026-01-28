import pandas as pd
import numpy as np
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from ..strategies.smart_exit import SmartExit

console = Console()

class BaseBacktester:
    """
    Modular Backtester Engine.
    Integrates with Feature Engineering and Smart Exit strategies.
    """
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        
    def run_backtest(
        self, 
        ticker_data: Dict[str, pd.DataFrame], 
        signals: Dict[str, pd.Series],
        exit_strategy: SmartExit = None
    ) -> pd.DataFrame:
        """
        Runs portfolio backtest.
        ticker_data: Dict mapping ticker -> DataFrame (with 'Close', 'SMA_50' etc)
        signals: Dict mapping ticker -> Series of 1 (Buy), -1 (Sell), 0 (Hold)
        """
        console.print("[bold]Running Backtest with Smart Exit...[/bold]")
        
        for ticker, df in ticker_data.items():
            if ticker not in signals:
                continue
                
            sig_series = signals[ticker]
            # Align signal index
            common_idx = df.index.intersection(sig_series.index)
            if len(common_idx) == 0:
                continue
                
            # Process Single Ticker
            self._process_ticker(ticker, df.loc[common_idx], sig_series.loc[common_idx], exit_strategy)
            
        return pd.DataFrame(self.trades)

    def _process_ticker(self, ticker: str, df: pd.DataFrame, signals: pd.Series, exit_strategy: SmartExit):
        position = 0
        entry_price = 0.0
        entry_idx = 0
        
        closes = df['Close'].values
        dates = df.index
        
        # Pre-calculate MA if needed for Smart Exit
        ma_series = None
        if exit_strategy and exit_strategy.moving_avg_period:
            # Assuming the DF has the MA column calculated, or we calc it on fly
            ma_col = f"SMA_{exit_strategy.moving_avg_period}"
            if ma_col in df.columns:
                ma_series = df[ma_col]
            else:
                ma_series = df['Close'].rolling(window=exit_strategy.moving_avg_period).mean()

        for i in range(len(df) - 1): # Stop 1 day early to handle 'Next Day' returns logic if needed
            date = dates[i]
            price = closes[i]
            signal = signals.iloc[i]
            
            # Check Exit First
            if position != 0 and exit_strategy:
                # Calculate max price since entry for trailing stop
                # For simplicity in this loop, we pass dynamic data
                # In robust implementation, we track 'highest_price' incrementally
                pass # (Logic simplified for prototype class structure)
                
            # Simplified Logic matching our Script
            if position == 0 and signal == 1: # Buy
                position = 1
                entry_price = price
                entry_idx = i
            
            elif position == 1:
                # Check Signal Exit or Smart Exit
                should_exit = False
                reason = ""
                
                # 1. Signal Flip
                if signal == -1: 
                    should_exit = True
                    reason = "Signal Flip"
                
                # 2. Smart Exit (Overrides Signal)
                if not should_exit and exit_strategy:
                    # Determine highest price so far
                    current_highest = np.max(closes[entry_idx : i+1])
                    days_held = i - entry_idx
                    should_exit, reason = exit_strategy.check_exit(
                        entry_price, price, current_highest, days_held, ma_series, i
                    )
                
                if should_exit:
                    pnl = (price - entry_price) / entry_price
                    self.trades.append({
                        "Ticker": ticker,
                        "Entry Date": dates[entry_idx],
                        "Exit Date": date,
                        "Return": pnl,
                        "Reason": reason
                    })
                    position = 0
                    
        # Force Close End
        if position == 1:
            price = closes[-1]
            pnl = (price - entry_price) / entry_price
            self.trades.append({
                "Ticker": ticker,
                "Entry Date": dates[entry_idx],
                "Exit Date": dates[-1],
                "Return": pnl,
                "Reason": "End of Data"
            })

    def summary(self):
        df = pd.DataFrame(self.trades)
        if df.empty:
            console.print("[red]No trades generated.[/red]")
            return
            
        avg_ret = df['Return'].mean()
        win_rate = (df['Return'] > 0).mean()
        
        table = Table(title="Backtest Summary (Smart Exit)")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Total Trades", str(len(df)))
        table.add_row("Avg Return", f"{avg_ret*100:.2f}%")
        table.add_row("Win Rate", f"{win_rate*100:.1f}%")
        console.print(table)
