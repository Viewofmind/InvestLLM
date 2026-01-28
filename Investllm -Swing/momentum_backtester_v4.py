"""
Momentum Strategy V4 - NO ML, Pure Price Action
Goal: Beat NIFTY 50 (~12% CAGR) with simple momentum rules

Entry Rules:
- Stock closes at 20-day high
- RSI > 50 (momentum confirmation)
- Volume > 20-day average (participation)

Exit Rules:
- 7-day max hold
- 5% stop loss
- 15% profit target
- Trailing stop at 8% profit

This serves as a baseline to compare against ML strategies.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MomentumStrategyV4:
    """
    Simple momentum strategy - no ML required

    Historically, momentum strategies return 15-20% CAGR in India
    """

    def __init__(
        self,
        lookback_high: int = 20,      # N-day high for entry
        rsi_period: int = 14,
        rsi_threshold: float = 50,     # RSI > 50 for entry
        volume_multiplier: float = 1.2,  # Volume > 1.2x average
        stop_loss: float = 0.05,       # 5% stop
        profit_target: float = 0.15,   # 15% target
        trailing_activation: float = 0.08,  # Activate trail at 8%
        trailing_distance: float = 0.04,    # 4% trail
        max_hold_days: int = 7,
        excluded_stocks: List[str] = None
    ):
        self.lookback_high = lookback_high
        self.rsi_period = rsi_period
        self.rsi_threshold = rsi_threshold
        self.volume_multiplier = volume_multiplier
        self.stop_loss = stop_loss
        self.profit_target = profit_target
        self.trailing_activation = trailing_activation
        self.trailing_distance = trailing_distance
        self.max_hold_days = max_hold_days
        self.excluded_stocks = excluded_stocks or ['ATGL', 'OFSS', 'ICICIGI', 'BERGEPAINT']

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50  # Default

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    def check_entry_signal(
        self,
        symbol: str,
        current_price: float,
        high_prices: pd.Series,
        close_prices: pd.Series,
        volumes: pd.Series
    ) -> Tuple[bool, Dict]:
        """
        Check if stock meets momentum entry criteria

        Returns: (should_enter, signal_info)
        """
        if symbol in self.excluded_stocks:
            return False, {}

        if len(close_prices) < self.lookback_high:
            return False, {}

        # 1. Check if at N-day high
        n_day_high = high_prices.iloc[-self.lookback_high:].max()
        at_high = current_price >= n_day_high * 0.98  # Within 2% of high

        if not at_high:
            return False, {}

        # 2. Check RSI > threshold
        rsi = self.calculate_rsi(close_prices, self.rsi_period)
        rsi_ok = rsi > self.rsi_threshold

        if not rsi_ok:
            return False, {}

        # 3. Check volume surge
        avg_volume = volumes.iloc[-20:].mean()
        current_volume = volumes.iloc[-1]
        volume_ok = current_volume > avg_volume * self.volume_multiplier

        if not volume_ok:
            return False, {}

        # All conditions met
        signal_info = {
            'rsi': rsi,
            'n_day_high': n_day_high,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1,
            'signal_strength': (rsi - 50) / 50  # 0-1 scale
        }

        return True, signal_info

    def check_exit_signal(
        self,
        entry_price: float,
        current_price: float,
        peak_price: float,
        days_held: int
    ) -> Tuple[bool, str]:
        """
        Check exit conditions

        Returns: (should_exit, reason)
        """
        profit_pct = (current_price - entry_price) / entry_price

        # 1. Profit target
        if profit_pct >= self.profit_target:
            return True, f"PROFIT_TARGET_{self.profit_target*100:.0f}%"

        # 2. Stop loss
        if profit_pct <= -self.stop_loss:
            return True, f"STOP_LOSS_{self.stop_loss*100:.0f}%"

        # 3. Trailing stop
        peak_profit = (peak_price - entry_price) / entry_price
        if peak_profit >= self.trailing_activation:
            trail_price = peak_price * (1 - self.trailing_distance)
            if current_price <= trail_price:
                return True, f"TRAILING_STOP"

        # 4. Max hold
        if days_held >= self.max_hold_days:
            return True, f"MAX_HOLD_{self.max_hold_days}D"

        return False, "HOLD"


class MomentumBacktesterV4:
    """
    Backtester for momentum strategy V4
    """

    def __init__(
        self,
        feature_dir: str,
        initial_capital: float = 100000,
        max_positions: int = 10
    ):
        self.feature_dir = feature_dir
        self.initial_capital = initial_capital
        self.max_positions = max_positions

        self.strategy = MomentumStrategyV4()
        self.trades = []
        self.equity_curve = []

    def load_data(self) -> pd.DataFrame:
        """Load price data from feature files"""
        feature_path = Path(self.feature_dir)

        # Try combined file
        combined_file = feature_path / 'all_swing_features.parquet'
        if combined_file.exists():
            df = pd.read_parquet(combined_file)
            print(f"Loaded: {len(df):,} rows")
            return df

        # Load individual files
        files = list(feature_path.glob('*_swing_features.parquet'))
        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(files)} files: {len(df):,} rows")
        return df

    def run_backtest(self) -> Dict:
        """Run momentum backtest"""
        print("\n" + "="*60)
        print("MOMENTUM STRATEGY V4 - NO ML")
        print("="*60)

        # Load data
        df = self.load_data()
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        # Get unique dates
        dates = sorted(df['timestamp'].unique())
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Trading days: {len(dates)}")

        # Initialize
        capital = self.initial_capital
        positions = {}  # symbol -> position info

        print(f"\nStarting backtest with ₹{capital:,.0f}...")
        print(f"Max positions: {self.max_positions}")
        print(f"Strategy: {self.strategy.lookback_high}-day high breakout")
        print("-"*60)

        # Process each day
        for day_idx, current_date in enumerate(dates):
            if day_idx < 30:  # Need history
                continue

            if day_idx % 500 == 0:
                print(f"Day {day_idx}/{len(dates)}: {current_date}, Equity: ₹{capital:,.0f}")

            day_df = df[df['timestamp'] == current_date]

            # Update existing positions
            for symbol in list(positions.keys()):
                position = positions[symbol]

                # Calculate days held
                days_held = (pd.Timestamp(current_date) - pd.Timestamp(position['entry_date'])).days

                symbol_row = day_df[day_df['symbol'] == symbol]

                # Force exit if no data
                if symbol_row.empty:
                    if days_held >= 10:
                        last_price = position.get('last_price', position['entry_price'])
                        profit_pct = (last_price - position['entry_price']) / position['entry_price']
                        profit = profit_pct * position['invested']
                        capital += position['invested'] + profit

                        self.trades.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': last_price,
                            'quantity': position['quantity'],
                            'profit_pct': profit_pct,
                            'profit_amount': profit,
                            'exit_reason': 'FORCE_EXIT_NO_DATA',
                            'days_held': days_held
                        })
                        del positions[symbol]
                    continue

                current_price = symbol_row['close'].values[0]
                position['last_price'] = current_price

                # Update peak
                if current_price > position['peak_price']:
                    position['peak_price'] = current_price

                # Check exit
                should_exit, reason = self.strategy.check_exit_signal(
                    position['entry_price'],
                    current_price,
                    position['peak_price'],
                    days_held
                )

                if should_exit:
                    profit_pct = (current_price - position['entry_price']) / position['entry_price']
                    profit = profit_pct * position['invested']
                    capital += position['invested'] + profit

                    self.trades.append({
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'profit_pct': profit_pct,
                        'profit_amount': profit,
                        'exit_reason': reason,
                        'days_held': days_held
                    })
                    del positions[symbol]

            # Look for new entries
            if len(positions) < self.max_positions:
                available = day_df[
                    (~day_df['symbol'].isin(positions.keys())) &
                    (~day_df['symbol'].isin(self.strategy.excluded_stocks))
                ]

                entry_signals = []

                for _, row in available.iterrows():
                    symbol = row['symbol']

                    # Get historical data
                    symbol_hist = df[
                        (df['symbol'] == symbol) &
                        (df['timestamp'] <= current_date)
                    ].tail(30)

                    if len(symbol_hist) < 25:
                        continue

                    should_enter, signal_info = self.strategy.check_entry_signal(
                        symbol,
                        row['close'],
                        symbol_hist['high'],
                        symbol_hist['close'],
                        symbol_hist['volume']
                    )

                    if should_enter:
                        entry_signals.append({
                            'symbol': symbol,
                            'price': row['close'],
                            'signal_strength': signal_info.get('signal_strength', 0.5),
                            'rsi': signal_info.get('rsi', 50)
                        })

                # Sort by signal strength, take top signals
                entry_signals.sort(key=lambda x: x['signal_strength'], reverse=True)

                for signal in entry_signals[:self.max_positions - len(positions)]:
                    symbol = signal['symbol']
                    entry_price = signal['price']

                    # Position sizing
                    position_value = capital / self.max_positions
                    if position_value < 1000:
                        continue

                    quantity = int(position_value / entry_price)
                    if quantity > 0:
                        invested = quantity * entry_price
                        capital -= invested

                        positions[symbol] = {
                            'entry_date': current_date,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'invested': invested,
                            'peak_price': entry_price,
                            'last_price': entry_price,
                            'rsi': signal['rsi']
                        }

            # Record equity
            position_value = sum(
                p['quantity'] * df[
                    (df['symbol'] == sym) &
                    (df['timestamp'] == current_date)
                ]['close'].values[0]
                for sym, p in positions.items()
                if len(df[(df['symbol'] == sym) & (df['timestamp'] == current_date)]) > 0
            )
            total_equity = capital + position_value

            self.equity_curve.append({
                'date': current_date,
                'cash': capital,
                'positions_value': position_value,
                'total_equity': total_equity,
                'num_positions': len(positions)
            })

        # Close remaining positions
        final_date = dates[-1]
        for symbol, position in list(positions.items()):
            symbol_row = df[(df['symbol'] == symbol) & (df['timestamp'] == final_date)]
            if not symbol_row.empty:
                exit_price = symbol_row['close'].values[0]
                profit_pct = (exit_price - position['entry_price']) / position['entry_price']
                profit = profit_pct * position['invested']
                capital += position['invested'] + profit

                self.trades.append({
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': final_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'profit_pct': profit_pct,
                    'profit_amount': profit,
                    'exit_reason': 'END_OF_BACKTEST',
                    'days_held': (pd.Timestamp(final_date) - pd.Timestamp(position['entry_date'])).days
                })

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        total_trades = len(trades_df)
        winners = trades_df[trades_df['profit_pct'] > 0]
        losers = trades_df[trades_df['profit_pct'] <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        # Returns
        total_profit = trades_df['profit_amount'].sum()
        avg_return = trades_df['profit_pct'].mean()
        total_return = (equity_df['total_equity'].iloc[-1] - self.initial_capital) / self.initial_capital

        # CAGR
        years = (pd.Timestamp(equity_df['date'].iloc[-1]) - pd.Timestamp(equity_df['date'].iloc[0])).days / 365.25
        if years > 0:
            cagr = ((equity_df['total_equity'].iloc[-1] / self.initial_capital) ** (1/years)) - 1
        else:
            cagr = 0

        # Risk metrics
        equity_df['daily_return'] = equity_df['total_equity'].pct_change()
        sharpe = equity_df['daily_return'].mean() / equity_df['daily_return'].std() * np.sqrt(252) if equity_df['daily_return'].std() > 0 else 0

        # Drawdown
        equity_df['peak'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()

        # Profit factor
        gross_profit = winners['profit_amount'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['profit_amount'].sum()) if len(losers) > 0 else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_hold_days': trades_df['days_held'].mean(),
            'final_equity': equity_df['total_equity'].iloc[-1]
        }

    def print_results(self, metrics: Dict):
        """Print results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS V4 - MOMENTUM STRATEGY")
        print("="*60)

        print(f"\nTotal Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Average Return: {metrics['avg_return']*100:.2f}%")
        print(f"Total Return: {metrics['total_return']*100:.2f}%")
        print(f"CAGR: {metrics['cagr']*100:.2f}%")
        print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"\nAvg Hold Days: {metrics['avg_hold_days']:.1f}")
        print(f"Final Equity: ₹{metrics['final_equity']:,.0f}")

        # Comparison with NIFTY
        print("\n" + "-"*60)
        print("COMPARISON WITH NIFTY 50")
        print("-"*60)
        nifty_cagr = 0.12  # ~12% historical
        print(f"NIFTY 50 CAGR (historical): ~{nifty_cagr*100:.0f}%")
        print(f"Our Strategy CAGR: {metrics['cagr']*100:.2f}%")

        if metrics['cagr'] > nifty_cagr:
            alpha = metrics['cagr'] - nifty_cagr
            print(f"ALPHA: +{alpha*100:.2f}% (BEATING NIFTY!)")
        else:
            shortfall = nifty_cagr - metrics['cagr']
            print(f"SHORTFALL: -{shortfall*100:.2f}% (underperforming)")

    def save_results(self, output_dir: str):
        """Save results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.trades:
            pd.DataFrame(self.trades).to_csv(output_path / 'all_trades_v4.csv', index=False)

        if self.equity_curve:
            pd.DataFrame(self.equity_curve).to_csv(output_path / 'equity_curve_v4.csv', index=False)

        # Exit analysis
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            exit_analysis = trades_df.groupby('exit_reason').agg({
                'symbol': 'count',
                'profit_pct': ['mean', lambda x: (x > 0).mean()],
                'profit_amount': 'sum'
            }).round(4)
            exit_analysis.columns = ['trades', 'avg_return', 'win_rate', 'total_profit']
            exit_analysis = exit_analysis.sort_values('total_profit', ascending=False)
            exit_analysis.to_csv(output_path / 'exit_reason_analysis_v4.csv')

        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Momentum Strategy V4 Backtest')
    parser.add_argument('--features', type=str, default='swing_features/')
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--output', type=str, default='reports/momentum_backtest_v4/')
    parser.add_argument('--max-positions', type=int, default=10)

    args = parser.parse_args()

    backtester = MomentumBacktesterV4(
        feature_dir=args.features,
        initial_capital=args.capital,
        max_positions=args.max_positions
    )

    metrics = backtester.run_backtest()
    backtester.print_results(metrics)
    backtester.save_results(args.output)


if __name__ == '__main__':
    main()
