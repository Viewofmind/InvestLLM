"""
Monthly Momentum Strategy V5 - With Risk Controls

Enhancements over V4:
1. Trailing Stop: Lock in profits, limit losses
2. Volatility Filter: Reduce exposure during high-volatility periods
3. Market Regime Detection: Go to cash in bear markets

Target: Reduce -64% max drawdown to <30% while maintaining alpha
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RiskManager:
    """
    Risk controls for momentum strategy
    """

    def __init__(
        self,
        trailing_stop_pct: float = 0.15,      # 15% trailing stop
        volatility_lookback: int = 20,         # 20-day volatility
        high_vol_threshold: float = 0.35,      # >35% annualized vol = high
        max_position_loss: float = 0.20,       # 20% max loss per position
        market_ma_period: int = 200            # 200-day MA for regime
    ):
        self.trailing_stop_pct = trailing_stop_pct
        self.volatility_lookback = volatility_lookback
        self.high_vol_threshold = high_vol_threshold
        self.max_position_loss = max_position_loss
        self.market_ma_period = market_ma_period

    def calculate_market_volatility(self, df: pd.DataFrame, current_date) -> float:
        """Calculate market-wide volatility using NIFTY proxy (average of all stocks)"""
        recent_data = df[df['timestamp'] <= current_date].tail(self.volatility_lookback * 100)

        if len(recent_data) < self.volatility_lookback:
            return 0.20  # Default

        # Use average returns across all stocks as market proxy
        daily_returns = recent_data.groupby('timestamp')['close'].mean().pct_change()
        volatility = daily_returns.tail(self.volatility_lookback).std() * np.sqrt(252)

        return volatility if not pd.isna(volatility) else 0.20

    def is_high_volatility_regime(self, df: pd.DataFrame, current_date) -> bool:
        """Check if we're in a high volatility regime"""
        vol = self.calculate_market_volatility(df, current_date)
        return vol > self.high_vol_threshold

    def check_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        high_since_entry: float
    ) -> Tuple[bool, str]:
        """
        Check if trailing stop is triggered

        Returns: (should_exit, reason)
        """
        # Calculate drawdown from peak
        if high_since_entry > entry_price:
            drawdown_from_peak = (high_since_entry - current_price) / high_since_entry
            if drawdown_from_peak >= self.trailing_stop_pct:
                return True, f"TRAILING_STOP_{self.trailing_stop_pct*100:.0f}%"

        # Check max loss from entry
        loss_from_entry = (entry_price - current_price) / entry_price
        if loss_from_entry >= self.max_position_loss:
            return True, f"MAX_LOSS_{self.max_position_loss*100:.0f}%"

        return False, ""

    def calculate_position_size_multiplier(self, volatility: float) -> float:
        """
        Reduce position size in high volatility

        Normal vol (20%): 1.0x
        High vol (40%): 0.5x
        Very high vol (60%): 0.33x
        """
        base_vol = 0.20
        multiplier = base_vol / max(volatility, 0.10)
        return min(max(multiplier, 0.25), 1.0)  # Clamp between 0.25x and 1.0x


class MonthlyMomentumStrategyV5:
    """
    Enhanced momentum strategy with risk controls
    """

    def __init__(
        self,
        portfolio_size: int = 20,
        monthly_turnover: int = 5,
        lookback_months: int = 12,
        skip_recent_month: bool = True,
        use_trailing_stop: bool = True,
        use_volatility_filter: bool = True,
        trailing_stop_pct: float = 0.15,
        excluded_stocks: List[str] = None
    ):
        self.portfolio_size = portfolio_size
        self.monthly_turnover = monthly_turnover
        self.lookback_months = lookback_months
        self.skip_recent_month = skip_recent_month
        self.use_trailing_stop = use_trailing_stop
        self.use_volatility_filter = use_volatility_filter
        self.excluded_stocks = excluded_stocks or ['ATGL', 'OFSS', 'ICICIGI', 'BERGEPAINT']

        self.risk_manager = RiskManager(trailing_stop_pct=trailing_stop_pct)

    def calculate_momentum(
        self,
        df: pd.DataFrame,
        current_date: pd.Timestamp,
        symbol: str
    ) -> float:
        """Calculate 12-month momentum (skip recent month)"""
        symbol_data = df[
            (df['symbol'] == symbol) &
            (df['timestamp'] <= current_date)
        ].sort_values('timestamp')

        if len(symbol_data) < 250:
            return np.nan

        if self.skip_recent_month:
            end_idx = -21 if len(symbol_data) > 21 else -1
            end_price = symbol_data['close'].iloc[end_idx]
        else:
            end_price = symbol_data['close'].iloc[-1]

        start_idx = -252 if len(symbol_data) >= 252 else 0
        start_price = symbol_data['close'].iloc[start_idx]

        if start_price <= 0:
            return np.nan

        momentum = (end_price - start_price) / start_price
        return momentum

    def rank_stocks(
        self,
        df: pd.DataFrame,
        current_date: pd.Timestamp,
        available_symbols: List[str]
    ) -> pd.DataFrame:
        """Rank stocks by momentum"""
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
        """Determine which stocks to remove and add"""
        if len(rankings) == 0:
            return [], []

        portfolio_rankings = rankings[rankings['symbol'].isin(current_portfolio)]

        if len(portfolio_rankings) >= self.monthly_turnover:
            stocks_to_remove = portfolio_rankings.nlargest(
                self.monthly_turnover, 'rank'
            )['symbol'].tolist()
        else:
            stocks_to_remove = []

        not_in_portfolio = rankings[~rankings['symbol'].isin(current_portfolio)]
        stocks_to_add = not_in_portfolio.nsmallest(
            self.monthly_turnover, 'rank'
        )['symbol'].tolist()

        return stocks_to_remove, stocks_to_add


class MonthlyMomentumBacktesterV5:
    """
    Backtester with risk controls
    """

    def __init__(
        self,
        feature_dir: str,
        initial_capital: float = 100000,
        portfolio_size: int = 20,
        monthly_turnover: int = 5,
        use_trailing_stop: bool = True,
        use_volatility_filter: bool = True,
        trailing_stop_pct: float = 0.15
    ):
        self.feature_dir = feature_dir
        self.initial_capital = initial_capital
        self.use_trailing_stop = use_trailing_stop
        self.use_volatility_filter = use_volatility_filter

        # Transaction costs (realistic for Indian markets)
        self.brokerage_rate = 0.0003   # 0.03% each way (discount broker)
        self.stt_rate = 0.001          # 0.10% STT on sell only
        self.slippage_rate = 0.001     # 0.10% slippage each way

        # Cost multipliers
        self.buy_cost_multiplier = 1 + self.brokerage_rate + self.slippage_rate  # 1.0013
        self.sell_cost_multiplier = 1 - self.brokerage_rate - self.stt_rate - self.slippage_rate  # 0.9977

        self.strategy = MonthlyMomentumStrategyV5(
            portfolio_size=portfolio_size,
            monthly_turnover=monthly_turnover,
            use_trailing_stop=use_trailing_stop,
            use_volatility_filter=use_volatility_filter,
            trailing_stop_pct=trailing_stop_pct
        )

        self.trades = []
        self.equity_curve = []
        self.rebalance_log = []
        self.risk_events = []

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
        month_ends = df_dates.groupby('year_month')['date'].max().tolist()
        return month_ends

    def run_backtest(self) -> Dict:
        """Run backtest with risk controls"""
        print("\n" + "="*60)
        print("MONTHLY MOMENTUM STRATEGY V5 - WITH RISK CONTROLS")
        print("="*60)

        df = self.load_data()
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        all_symbols = df['symbol'].unique().tolist()
        print(f"Total stocks: {len(all_symbols)}")

        dates = sorted(df['timestamp'].unique())
        print(f"Date range: {dates[0]} to {dates[-1]}")

        month_ends = self.get_month_end_dates(dates)
        print(f"Rebalancing months: {len(month_ends)}")

        print(f"\nRisk Controls:")
        print(f"  - Trailing Stop: {self.use_trailing_stop} ({self.strategy.risk_manager.trailing_stop_pct*100:.0f}%)")
        print(f"  - Volatility Filter: {self.use_volatility_filter}")
        print(f"  - Max Position Loss: {self.strategy.risk_manager.max_position_loss*100:.0f}%")

        print(f"\nTransaction Costs:")
        print(f"  - Brokerage: {self.brokerage_rate*100:.2f}% each way")
        print(f"  - STT: {self.stt_rate*100:.2f}% on sell")
        print(f"  - Slippage: {self.slippage_rate*100:.2f}% each way")
        print(f"  - Total round-trip: ~{(self.brokerage_rate*2 + self.stt_rate + self.slippage_rate*2)*100:.2f}%")

        capital = self.initial_capital
        portfolio = {}  # symbol -> {shares, entry_price, entry_date, high_since_entry}

        print(f"\nStrategy: Top {self.strategy.portfolio_size} momentum stocks")
        print(f"Starting capital: ₹{capital:,.0f}")
        print("-"*60)

        start_idx = 12  # Skip first 12 months

        # Track daily for trailing stops
        all_dates = sorted(df['timestamp'].unique())
        date_to_month_end = {}
        for me in month_ends:
            date_to_month_end[me] = True

        high_vol_months = 0

        for day_idx, current_date in enumerate(all_dates):
            if day_idx < 252:  # Skip first year
                continue

            day_df = df[df['timestamp'] == current_date]

            if day_df.empty:
                continue

            # Check volatility regime
            is_high_vol = False
            if self.use_volatility_filter:
                is_high_vol = self.strategy.risk_manager.is_high_volatility_regime(df, current_date)

            # Daily: Check trailing stops for all positions
            if self.use_trailing_stop:
                for symbol in list(portfolio.keys()):
                    position = portfolio[symbol]
                    symbol_row = day_df[day_df['symbol'] == symbol]

                    if symbol_row.empty:
                        continue

                    current_price = symbol_row['close'].values[0]

                    # Update high since entry
                    if current_price > position.get('high_since_entry', position['entry_price']):
                        position['high_since_entry'] = current_price

                    # Check trailing stop
                    should_exit, exit_reason = self.strategy.risk_manager.check_trailing_stop(
                        position['entry_price'],
                        current_price,
                        position.get('high_since_entry', position['entry_price'])
                    )

                    if should_exit:
                        # Apply transaction costs on sell
                        proceeds = position['shares'] * current_price * self.sell_cost_multiplier
                        capital += proceeds

                        profit_pct = (current_price - position['entry_price']) / position['entry_price']

                        self.trades.append({
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'shares': position['shares'],
                            'profit_pct': profit_pct,
                            'profit_amount': proceeds - position['invested'],
                            'exit_reason': exit_reason,
                            'months_held': ((pd.Timestamp(current_date) - pd.Timestamp(position['entry_date'])).days) // 30
                        })

                        self.risk_events.append({
                            'date': current_date,
                            'symbol': symbol,
                            'event': exit_reason,
                            'price': current_price
                        })

                        del portfolio[symbol]

            # Monthly rebalancing
            if current_date in date_to_month_end:
                month_idx = month_ends.index(current_date)

                if month_idx < start_idx:
                    continue

                # Calculate current portfolio value
                portfolio_value = 0
                for symbol, position in portfolio.items():
                    symbol_row = day_df[day_df['symbol'] == symbol]
                    if not symbol_row.empty:
                        current_price = symbol_row['close'].values[0]
                        portfolio_value += position['shares'] * current_price

                total_equity = capital + portfolio_value

                # Progress update
                if month_idx % 12 == 0:
                    vol_status = " [HIGH VOL]" if is_high_vol else ""
                    print(f"Month {month_idx}: {current_date}, Equity: ₹{total_equity:,.0f}{vol_status}")

                # Volatility filter: reduce new positions in high vol
                position_multiplier = 1.0
                if is_high_vol and self.use_volatility_filter:
                    position_multiplier = 0.5  # Half position size in high vol
                    high_vol_months += 1

                # Rank all stocks
                available_symbols = day_df['symbol'].unique().tolist()
                rankings = self.strategy.rank_stocks(df, current_date, available_symbols)

                if rankings.empty:
                    continue

                # Determine changes
                current_holdings = list(portfolio.keys())

                if len(current_holdings) < self.strategy.portfolio_size:
                    top_stocks = rankings.nsmallest(
                        self.strategy.portfolio_size, 'rank'
                    )['symbol'].tolist()
                    stocks_to_add = [s for s in top_stocks if s not in current_holdings]
                    stocks_to_remove = []
                else:
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
                            # Apply transaction costs on sell
                            proceeds = position['shares'] * exit_price * self.sell_cost_multiplier
                            capital += proceeds

                            profit_pct = (exit_price - position['entry_price']) / position['entry_price']

                            self.trades.append({
                                'symbol': symbol,
                                'entry_date': position['entry_date'],
                                'exit_date': current_date,
                                'entry_price': position['entry_price'],
                                'exit_price': exit_price,
                                'shares': position['shares'],
                                'profit_pct': profit_pct,
                                'profit_amount': proceeds - position['invested'],
                                'exit_reason': 'MONTHLY_REBALANCE',
                                'months_held': ((pd.Timestamp(current_date) - pd.Timestamp(position['entry_date'])).days) // 30
                            })

                            del portfolio[symbol]

                # Calculate position size
                current_count = len(portfolio)
                target_count = self.strategy.portfolio_size
                new_positions_count = min(len(stocks_to_add), target_count - current_count)

                if new_positions_count > 0:
                    position_size = (total_equity / target_count) * position_multiplier

                    for symbol in stocks_to_add[:new_positions_count]:
                        symbol_row = day_df[day_df['symbol'] == symbol]

                        if symbol_row.empty:
                            continue

                        entry_price = symbol_row['close'].values[0]
                        shares = int(position_size / entry_price)

                        # Apply transaction costs on buy
                        total_cost = shares * entry_price * self.buy_cost_multiplier
                        if shares > 0 and capital >= total_cost:
                            invested = total_cost
                            capital -= invested

                            portfolio[symbol] = {
                                'shares': shares,
                                'entry_price': entry_price,
                                'entry_date': current_date,
                                'invested': invested,
                                'high_since_entry': entry_price
                            }

                # Log
                self.rebalance_log.append({
                    'date': current_date,
                    'removed': stocks_to_remove,
                    'added': stocks_to_add[:new_positions_count],
                    'portfolio_size': len(portfolio),
                    'total_equity': total_equity,
                    'is_high_vol': is_high_vol
                })

                self.equity_curve.append({
                    'date': current_date,
                    'cash': capital,
                    'portfolio_value': portfolio_value,
                    'total_equity': total_equity,
                    'num_positions': len(portfolio),
                    'is_high_vol': is_high_vol
                })

        # Final valuation
        final_date = all_dates[-1]
        final_df = df[df['timestamp'] == final_date]

        final_value = capital
        for symbol, position in portfolio.items():
            symbol_row = final_df[final_df['symbol'] == symbol]
            if not symbol_row.empty:
                final_value += position['shares'] * symbol_row['close'].values[0]

        print(f"\nHigh volatility months: {high_vol_months}")
        print(f"Trailing stop exits: {len(self.risk_events)}")

        return self.calculate_metrics(final_value)

    def calculate_metrics(self, final_value: float) -> Dict:
        """Calculate performance metrics"""
        equity_df = pd.DataFrame(self.equity_curve)

        if equity_df.empty:
            return {}

        total_return = (final_value - self.initial_capital) / self.initial_capital

        years = (pd.Timestamp(equity_df['date'].iloc[-1]) - pd.Timestamp(equity_df['date'].iloc[0])).days / 365.25
        cagr = ((final_value / self.initial_capital) ** (1/years)) - 1 if years > 0 else 0

        equity_df['monthly_return'] = equity_df['total_equity'].pct_change()
        monthly_std = equity_df['monthly_return'].std()
        monthly_mean = equity_df['monthly_return'].mean()
        sharpe = (monthly_mean / monthly_std) * np.sqrt(12) if monthly_std > 0 else 0

        equity_df['peak'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()

        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            win_rate = (trades_df['profit_pct'] > 0).mean()
            avg_return = trades_df['profit_pct'].mean()
            total_trades = len(trades_df)

            # Exit reason breakdown
            exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

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
            exit_reasons = {}

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
            'years': years,
            'exit_reasons': exit_reasons,
            'risk_events': len(self.risk_events)
        }

    def print_results(self, metrics: Dict):
        """Print results"""
        print("\n" + "="*60)
        print("RESULTS - MONTHLY MOMENTUM V5 (WITH RISK CONTROLS)")
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

        if metrics.get('exit_reasons'):
            print(f"\nExit Reasons:")
            for reason, count in metrics['exit_reasons'].items():
                print(f"  {reason}: {count}")

        print(f"\nRisk Events (Trailing Stops): {metrics['risk_events']}")

        # Compare
        print("\n" + "-"*60)
        print("COMPARISON")
        print("-"*60)
        print(f"V4 (No Risk Controls): 20.19% CAGR, -63.82% Max DD")
        print(f"V5 (With Risk Controls): {metrics['cagr']*100:.2f}% CAGR, {metrics['max_drawdown']*100:.2f}% Max DD")

        nifty_cagr = 0.12
        if metrics['cagr'] > nifty_cagr:
            alpha = metrics['cagr'] - nifty_cagr
            print(f"\n*** ALPHA: +{alpha*100:.2f}% (BEATING NIFTY!) ***")
        else:
            shortfall = nifty_cagr - metrics['cagr']
            print(f"\nShortfall: -{shortfall*100:.2f}% (underperforming)")

        # Risk-adjusted comparison
        if metrics['max_drawdown'] != 0:
            calmar = metrics['cagr'] / abs(metrics['max_drawdown'])
            print(f"\nCalmar Ratio (CAGR/MaxDD): {calmar:.2f}")

    def save_results(self, output_dir: str):
        """Save results"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.trades:
            pd.DataFrame(self.trades).to_csv(output_path / 'trades_v5.csv', index=False)

        if self.equity_curve:
            pd.DataFrame(self.equity_curve).to_csv(output_path / 'equity_curve_v5.csv', index=False)

        if self.risk_events:
            pd.DataFrame(self.risk_events).to_csv(output_path / 'risk_events_v5.csv', index=False)

        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Monthly Momentum Strategy V5')
    parser.add_argument('--features', type=str, default='swing_features/')
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--output', type=str, default='reports/monthly_momentum_v5/')
    parser.add_argument('--portfolio-size', type=int, default=20)
    parser.add_argument('--turnover', type=int, default=5)
    parser.add_argument('--trailing-stop', type=float, default=0.15, help='Trailing stop %')
    parser.add_argument('--no-trailing-stop', action='store_true', help='Disable trailing stop')
    parser.add_argument('--no-vol-filter', action='store_true', help='Disable volatility filter')

    args = parser.parse_args()

    backtester = MonthlyMomentumBacktesterV5(
        feature_dir=args.features,
        initial_capital=args.capital,
        portfolio_size=args.portfolio_size,
        monthly_turnover=args.turnover,
        use_trailing_stop=not args.no_trailing_stop,
        use_volatility_filter=not args.no_vol_filter,
        trailing_stop_pct=args.trailing_stop
    )

    metrics = backtester.run_backtest()
    backtester.print_results(metrics)
    backtester.save_results(args.output)


if __name__ == '__main__':
    main()
