"""
Monthly Momentum Rebalancing Strategy V4

Strategy:
- Hold top 20 stocks by momentum (12-month return)
- Every month: Remove 5 weakest, Add 5 strongest new stocks
- Equal weight all positions
- Rebalance monthly

This is a proven strategy that historically generates 15-20% CAGR
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MonthlyMomentumStrategy:
    """
    Monthly rebalancing momentum portfolio

    Ranks stocks by past returns and holds top performers
    """

    def __init__(
        self,
        portfolio_size: int = 20,          # Hold top 20 stocks
        monthly_turnover: int = 5,          # Replace 5 stocks per month
        lookback_months: int = 12,          # 12-month momentum
        skip_recent_month: bool = True,     # Skip last month (reversal effect)
        excluded_stocks: List[str] = None
    ):
        self.portfolio_size = portfolio_size
        self.monthly_turnover = monthly_turnover
        self.lookback_months = lookback_months
        self.skip_recent_month = skip_recent_month
        self.excluded_stocks = excluded_stocks or ['ATGL', 'OFSS', 'ICICIGI', 'BERGEPAINT']

    def calculate_momentum(
        self,
        df: pd.DataFrame,
        current_date: pd.Timestamp,
        symbol: str
    ) -> float:
        """
        Calculate momentum score for a stock

        Uses 12-month return, skipping most recent month
        (to avoid short-term reversal effect)
        """
        # Get historical data for this symbol
        symbol_data = df[
            (df['symbol'] == symbol) &
            (df['timestamp'] <= current_date)
        ].sort_values('timestamp')

        if len(symbol_data) < 250:  # Need ~1 year of data
            return np.nan

        # Get prices
        if self.skip_recent_month:
            # Skip last 21 trading days (~1 month)
            end_idx = -21 if len(symbol_data) > 21 else -1
            end_price = symbol_data['close'].iloc[end_idx]
        else:
            end_price = symbol_data['close'].iloc[-1]

        # Start price (12 months ago = ~252 trading days)
        start_idx = -252 if len(symbol_data) >= 252 else 0
        start_price = symbol_data['close'].iloc[start_idx]

        if start_price <= 0:
            return np.nan

        # Momentum = 12-month return
        momentum = (end_price - start_price) / start_price

        return momentum

    def rank_stocks(
        self,
        df: pd.DataFrame,
        current_date: pd.Timestamp,
        available_symbols: List[str]
    ) -> pd.DataFrame:
        """
        Rank all stocks by momentum

        Returns: DataFrame with symbol, momentum, rank
        """
        momentum_scores = []

        for symbol in available_symbols:
            if symbol in self.excluded_stocks:
                continue

            momentum = self.calculate_momentum(df, current_date, symbol)

            if not pd.isna(momentum):
                momentum_scores.append({
                    'symbol': symbol,
                    'momentum': momentum
                })

        # Create DataFrame and rank
        rankings = pd.DataFrame(momentum_scores)

        if len(rankings) == 0:
            return pd.DataFrame()

        rankings = rankings.sort_values('momentum', ascending=False)
        rankings['rank'] = range(1, len(rankings) + 1)

        return rankings

    def select_portfolio_changes(
        self,
        current_portfolio: List[str],
        rankings: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """
        Determine which stocks to remove and add

        Returns: (stocks_to_remove, stocks_to_add)
        """
        if len(rankings) == 0:
            return [], []

        # Get current portfolio rankings
        portfolio_rankings = rankings[rankings['symbol'].isin(current_portfolio)]

        # Find worst 5 in current portfolio
        if len(portfolio_rankings) >= self.monthly_turnover:
            stocks_to_remove = portfolio_rankings.nlargest(
                self.monthly_turnover, 'rank'
            )['symbol'].tolist()
        else:
            stocks_to_remove = []

        # Find top stocks not in portfolio
        not_in_portfolio = rankings[~rankings['symbol'].isin(current_portfolio)]
        stocks_to_add = not_in_portfolio.nsmallest(
            self.monthly_turnover, 'rank'
        )['symbol'].tolist()

        return stocks_to_remove, stocks_to_add


class MonthlyMomentumBacktester:
    """
    Backtester for monthly momentum rebalancing strategy
    """

    def __init__(
        self,
        feature_dir: str,
        initial_capital: float = 100000,
        portfolio_size: int = 20,
        monthly_turnover: int = 5
    ):
        self.feature_dir = feature_dir
        self.initial_capital = initial_capital

        self.strategy = MonthlyMomentumStrategy(
            portfolio_size=portfolio_size,
            monthly_turnover=monthly_turnover
        )

        self.trades = []
        self.equity_curve = []
        self.rebalance_log = []

    def load_data(self) -> pd.DataFrame:
        """Load price data"""
        feature_path = Path(self.feature_dir)
        combined_file = feature_path / 'all_swing_features.parquet'

        if combined_file.exists():
            df = pd.read_parquet(combined_file)
            print(f"Loaded: {len(df):,} rows")
            return df

        raise FileNotFoundError(f"Feature file not found: {combined_file}")

    def get_month_end_dates(self, dates: List) -> List:
        """Get last trading day of each month"""
        df_dates = pd.DataFrame({'date': dates})
        df_dates['date'] = pd.to_datetime(df_dates['date'])
        df_dates['year_month'] = df_dates['date'].dt.to_period('M')

        # Get last date of each month
        month_ends = df_dates.groupby('year_month')['date'].max().tolist()

        return month_ends

    def run_backtest(self) -> Dict:
        """Run monthly momentum backtest"""
        print("\n" + "="*60)
        print("MONTHLY MOMENTUM REBALANCING STRATEGY V4")
        print("="*60)

        # Load data
        df = self.load_data()
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        # Get all symbols
        all_symbols = df['symbol'].unique().tolist()
        print(f"Total stocks: {len(all_symbols)}")

        # Get unique dates
        dates = sorted(df['timestamp'].unique())
        print(f"Date range: {dates[0]} to {dates[-1]}")

        # Get month-end dates for rebalancing
        month_ends = self.get_month_end_dates(dates)
        print(f"Rebalancing months: {len(month_ends)}")

        # Initialize
        capital = self.initial_capital
        portfolio = {}  # symbol -> {shares, entry_price, entry_date}

        print(f"\nStrategy: Top {self.strategy.portfolio_size} momentum stocks")
        print(f"Monthly turnover: {self.strategy.monthly_turnover} stocks")
        print(f"Starting capital: ₹{capital:,.0f}")
        print("-"*60)

        # Skip first year (need 12 months for momentum calculation)
        # Note: start_idx is in MONTHS, not trading days
        start_idx = 12  # Skip first 12 months to have momentum history

        for month_idx, rebalance_date in enumerate(month_ends[start_idx:], start_idx):
            # Get current prices
            day_df = df[df['timestamp'] == rebalance_date]

            if day_df.empty:
                continue

            # Calculate current portfolio value
            portfolio_value = 0
            for symbol, position in portfolio.items():
                symbol_row = day_df[day_df['symbol'] == symbol]
                if not symbol_row.empty:
                    current_price = symbol_row['close'].values[0]
                    portfolio_value += position['shares'] * current_price

            total_equity = capital + portfolio_value

            # Print progress
            if month_idx % 12 == 0:
                print(f"Month {month_idx}: {rebalance_date}, Equity: ₹{total_equity:,.0f}")

            # Rank all stocks by momentum
            available_symbols = day_df['symbol'].unique().tolist()
            rankings = self.strategy.rank_stocks(df, rebalance_date, available_symbols)

            if rankings.empty:
                continue

            # Determine changes
            current_holdings = list(portfolio.keys())

            if len(current_holdings) < self.strategy.portfolio_size:
                # Initial build: add top stocks
                top_stocks = rankings.nsmallest(
                    self.strategy.portfolio_size, 'rank'
                )['symbol'].tolist()
                stocks_to_add = [s for s in top_stocks if s not in current_holdings]
                stocks_to_remove = []
            else:
                # Monthly rebalance
                stocks_to_remove, stocks_to_add = self.strategy.select_portfolio_changes(
                    current_holdings, rankings
                )

            # Execute sells
            for symbol in stocks_to_remove:
                if symbol in portfolio:
                    position = portfolio[symbol]
                    symbol_row = day_df[day_df['symbol'] == symbol]

                    if not symbol_row.empty:
                        exit_price = symbol_row['close'].values[0]
                        proceeds = position['shares'] * exit_price
                        capital += proceeds

                        profit_pct = (exit_price - position['entry_price']) / position['entry_price']

                        self.trades.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': rebalance_date,
                            'entry_price': position['entry_price'],
                            'exit_price': exit_price,
                            'shares': position['shares'],
                            'profit_pct': profit_pct,
                            'profit_amount': proceeds - position['invested'],
                            'exit_reason': 'MONTHLY_REBALANCE',
                            'months_held': ((pd.Timestamp(rebalance_date) - pd.Timestamp(position['entry_date'])).days) // 30
                        })

                        del portfolio[symbol]

            # Calculate position size for new stocks
            current_count = len(portfolio)
            target_count = self.strategy.portfolio_size
            new_positions_count = min(len(stocks_to_add), target_count - current_count)

            if new_positions_count > 0:
                # Equal weight
                position_size = total_equity / target_count

                # Execute buys
                for symbol in stocks_to_add[:new_positions_count]:
                    symbol_row = day_df[day_df['symbol'] == symbol]

                    if symbol_row.empty:
                        continue

                    entry_price = symbol_row['close'].values[0]
                    shares = int(position_size / entry_price)

                    if shares > 0 and capital >= shares * entry_price:
                        invested = shares * entry_price
                        capital -= invested

                        portfolio[symbol] = {
                            'shares': shares,
                            'entry_price': entry_price,
                            'entry_date': rebalance_date,
                            'invested': invested
                        }

            # Log rebalance
            self.rebalance_log.append({
                'date': rebalance_date,
                'removed': stocks_to_remove,
                'added': stocks_to_add[:new_positions_count],
                'portfolio_size': len(portfolio),
                'total_equity': total_equity
            })

            # Record equity
            self.equity_curve.append({
                'date': rebalance_date,
                'cash': capital,
                'portfolio_value': portfolio_value,
                'total_equity': total_equity,
                'num_positions': len(portfolio)
            })

        # Final valuation
        final_date = month_ends[-1]
        final_df = df[df['timestamp'] == final_date]

        final_value = capital
        for symbol, position in portfolio.items():
            symbol_row = final_df[final_df['symbol'] == symbol]
            if not symbol_row.empty:
                final_value += position['shares'] * symbol_row['close'].values[0]

        return self.calculate_metrics(final_value)

    def calculate_metrics(self, final_value: float) -> Dict:
        """Calculate performance metrics"""
        equity_df = pd.DataFrame(self.equity_curve)

        if equity_df.empty:
            return {}

        total_return = (final_value - self.initial_capital) / self.initial_capital

        # CAGR
        years = (pd.Timestamp(equity_df['date'].iloc[-1]) - pd.Timestamp(equity_df['date'].iloc[0])).days / 365.25
        cagr = ((final_value / self.initial_capital) ** (1/years)) - 1 if years > 0 else 0

        # Monthly returns for Sharpe
        equity_df['monthly_return'] = equity_df['total_equity'].pct_change()
        monthly_std = equity_df['monthly_return'].std()
        monthly_mean = equity_df['monthly_return'].mean()
        sharpe = (monthly_mean / monthly_std) * np.sqrt(12) if monthly_std > 0 else 0

        # Drawdown
        equity_df['peak'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()

        # Trade stats
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            win_rate = (trades_df['profit_pct'] > 0).mean()
            avg_return = trades_df['profit_pct'].mean()
            total_trades = len(trades_df)
            winners = trades_df[trades_df['profit_pct'] > 0]
            losers = trades_df[trades_df['profit_pct'] <= 0]
            gross_profit = winners['profit_amount'].sum() if len(winners) > 0 else 0
            gross_loss = abs(losers['profit_amount'].sum()) if len(losers) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_return = 0
            total_trades = 0
            profit_factor = 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'total_return': total_return,
            'cagr': cagr,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'final_equity': final_value,
            'years': years
        }

    def print_results(self, metrics: Dict):
        """Print results"""
        print("\n" + "="*60)
        print("RESULTS - MONTHLY MOMENTUM STRATEGY")
        print("="*60)

        print(f"\nTotal Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"Avg Trade Return: {metrics['avg_return']*100:.2f}%")

        print(f"\nTotal Return: {metrics['total_return']*100:.2f}%")
        print(f"CAGR: {metrics['cagr']*100:.2f}%")
        print(f"Years: {metrics['years']:.1f}")

        print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\nFinal Equity: ₹{metrics['final_equity']:,.0f}")

        # Compare with NIFTY
        print("\n" + "-"*60)
        print("COMPARISON WITH NIFTY 50")
        print("-"*60)
        nifty_cagr = 0.12
        print(f"NIFTY 50 CAGR (historical): ~{nifty_cagr*100:.0f}%")
        print(f"Our Strategy CAGR: {metrics['cagr']*100:.2f}%")

        if metrics['cagr'] > nifty_cagr:
            alpha = metrics['cagr'] - nifty_cagr
            print(f"\n*** ALPHA: +{alpha*100:.2f}% (BEATING NIFTY!) ***")
        else:
            shortfall = nifty_cagr - metrics['cagr']
            print(f"\nShortfall: -{shortfall*100:.2f}% (underperforming)")

    def save_results(self, output_dir: str):
        """Save results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.trades:
            pd.DataFrame(self.trades).to_csv(output_path / 'trades_monthly_momentum.csv', index=False)

        if self.equity_curve:
            pd.DataFrame(self.equity_curve).to_csv(output_path / 'equity_curve_monthly.csv', index=False)

        if self.rebalance_log:
            # Convert lists to strings for CSV
            log_df = pd.DataFrame(self.rebalance_log)
            log_df['removed'] = log_df['removed'].apply(lambda x: ','.join(x) if x else '')
            log_df['added'] = log_df['added'].apply(lambda x: ','.join(x) if x else '')
            log_df.to_csv(output_path / 'rebalance_log.csv', index=False)

        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Monthly Momentum Strategy')
    parser.add_argument('--features', type=str, default='swing_features/')
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--output', type=str, default='reports/monthly_momentum_v4/')
    parser.add_argument('--portfolio-size', type=int, default=20)
    parser.add_argument('--turnover', type=int, default=5)

    args = parser.parse_args()

    backtester = MonthlyMomentumBacktester(
        feature_dir=args.features,
        initial_capital=args.capital,
        portfolio_size=args.portfolio_size,
        monthly_turnover=args.turnover
    )

    metrics = backtester.run_backtest()
    backtester.print_results(metrics)
    backtester.save_results(args.output)


if __name__ == '__main__':
    main()
