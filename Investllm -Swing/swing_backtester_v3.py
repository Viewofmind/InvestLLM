"""
Swing Trading Backtester V3 - FULLY OPTIMIZED with GPU Support

Key Features:
1. Temperature scaling for confidence calibration
2. Ensemble model support (multiple models)
3. Partial profit taking
4. Volatility-based position sizing
5. GPU-accelerated inference (fast backtest)
6. Loosened stop losses
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from swing_exit_strategy_v3 import SwingExitStrategyV3, TradeConfigV3, TradeMonitorV3
from ensemble_predictor import TemperatureScaler, EnsemblePredictor


class SwingLSTM(nn.Module):
    """LSTM model for swing trading"""

    def __init__(self, input_dim, hidden_dim=256, num_layers=3, dropout=0.3):
        super(SwingLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze()


class SwingBacktesterV3:
    """
    V3 Backtester with all optimizations

    Supports:
    - Single model or ensemble
    - Temperature scaling
    - Partial exits
    - GPU acceleration
    """

    def __init__(
        self,
        model_paths: List[str],
        scaler_path: str,
        feature_dir: str,
        initial_capital: float = 100000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        temperature: float = 0.5,
        use_ensemble: bool = True
    ):
        self.model_paths = model_paths if isinstance(model_paths, list) else [model_paths]
        self.scaler_path = scaler_path
        self.feature_dir = feature_dir
        self.initial_capital = initial_capital
        self.device = device
        self.temperature = temperature
        self.use_ensemble = use_ensemble and len(self.model_paths) > 1

        # Strategy
        self.config = TradeConfigV3()
        self.strategy = SwingExitStrategyV3(self.config)

        # Temperature scaler
        self.temp_scaler = TemperatureScaler(temperature)

        # Results storage
        self.trades = []
        self.partial_exits = []
        self.equity_curve = []
        self.daily_returns = []

        print(f"Device: {self.device}")
        print(f"Temperature: {self.temperature}")
        print(f"Ensemble mode: {self.use_ensemble} ({len(self.model_paths)} models)")

    def load_models(self, input_dim: int) -> bool:
        """Load model(s) for inference"""
        self.models = []

        for path in self.model_paths:
            if not Path(path).exists():
                print(f"Model not found: {path}")
                continue

            try:
                checkpoint = torch.load(path, map_location=self.device)
                model = SwingLSTM(input_dim=input_dim)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(self.device)
                model.eval()
                self.models.append(model)
                print(f"Loaded: {path}")
            except Exception as e:
                print(f"Error loading {path}: {e}")

        return len(self.models) > 0

    def load_scaler(self) -> bool:
        """Load feature scaler"""
        if not Path(self.scaler_path).exists():
            print(f"Scaler not found: {self.scaler_path}")
            return False

        with open(self.scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"Loaded scaler: {self.scaler_path}")
        return True

    def load_features(self) -> pd.DataFrame:
        """Load all feature files"""
        feature_path = Path(self.feature_dir)

        # Try combined file first
        combined_file = feature_path / 'all_swing_features.parquet'
        if combined_file.exists():
            df = pd.read_parquet(combined_file)
            print(f"Loaded combined features: {len(df):,} rows")
            return df

        # Load individual files
        files = list(feature_path.glob('*_swing_features.parquet'))
        if not files:
            raise FileNotFoundError(f"No feature files in {feature_path}")

        dfs = [pd.read_parquet(f) for f in files]
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(files)} feature files: {len(df):,} rows")
        return df

    @torch.no_grad()
    def predict_batch(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated batch prediction

        Returns: (predictions, confidences)
        """
        # Convert to tensor
        X = torch.FloatTensor(features).to(self.device)

        if self.use_ensemble and len(self.models) > 1:
            # Ensemble prediction
            all_preds = []
            for model in self.models:
                pred = model(X)
                pred = torch.sigmoid(pred).cpu().numpy()
                all_preds.append(pred)

            all_preds = np.array(all_preds)
            mean_pred = np.mean(all_preds, axis=0)
            std_pred = np.std(all_preds, axis=0)

            # Calibrate
            calibrated = self.temp_scaler.scale(mean_pred)

            # Confidence based on agreement
            agreement = 1 - np.clip(std_pred * 5, 0, 0.5)
            base_conf = np.abs(calibrated - 0.5) * 2
            confidence = np.clip(base_conf * agreement + 0.2, 0, 1)

        else:
            # Single model
            pred = self.models[0](X)
            pred = torch.sigmoid(pred).cpu().numpy()

            # Calibrate
            calibrated = self.temp_scaler.scale(pred)

            # Confidence from prediction strength
            confidence = np.abs(calibrated - 0.5) * 2

        # Ensure always returns arrays (not scalars)
        calibrated = np.atleast_1d(calibrated)
        confidence = np.atleast_1d(confidence)

        return calibrated, confidence

    def _calculate_volatility(self, df_symbol: pd.DataFrame, idx: int, window: int = 20) -> float:
        """Calculate historical volatility"""
        if idx < window:
            return 0.30  # Default

        returns = df_symbol['close'].pct_change().iloc[idx-window:idx]
        return returns.std() * np.sqrt(252)  # Annualized

    def _calculate_atr_pct(self, df_symbol: pd.DataFrame, idx: int, window: int = 14) -> float:
        """Calculate ATR as percentage of price"""
        if idx < window:
            return 0.02

        high = df_symbol['high'].iloc[idx-window:idx]
        low = df_symbol['low'].iloc[idx-window:idx]
        close = df_symbol['close'].iloc[idx-window:idx]

        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        atr = tr.mean()
        return atr / df_symbol['close'].iloc[idx]

    def run_backtest(self, max_positions: int = 10, sequence_length: int = 30):
        """Run full backtest with V3 optimizations"""
        print("\n" + "="*60)
        print("SWING BACKTESTER V3 - FULLY OPTIMIZED")
        print("="*60)

        # Load data
        df = self.load_features()

        # Identify feature columns
        exclude_cols = ['symbol', 'timestamp', 'target', 'open', 'high', 'low', 'close', 'volume', 'exchange']
        feature_cols = [col for col in df.columns if col not in exclude_cols
                       and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        print(f"Features: {len(feature_cols)}")

        # Load models
        if not self.load_models(len(feature_cols)):
            raise ValueError("Failed to load models")
        if not self.load_scaler():
            raise ValueError("Failed to load scaler")

        # Sort by date
        df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)

        # Get unique dates
        dates = sorted(df['timestamp'].unique())
        print(f"Date range: {dates[0]} to {dates[-1]}")
        print(f"Trading days: {len(dates)}")

        # Initialize
        capital = self.initial_capital
        positions = {}  # symbol -> position info
        trade_monitor = TradeMonitorV3(self.strategy)

        # Tracking
        self.equity_curve = []
        self.trades = []
        trade_id_counter = 0

        print(f"\nStarting backtest with ₹{capital:,.0f}...")
        print(f"Max positions: {max_positions}")
        print(f"Temperature: {self.temperature}")
        print("-"*60)

        # Pre-compute features for speed
        print("Pre-computing scaled features...")
        df_features = df[feature_cols].values
        df_scaled = self.scaler.transform(df_features)

        # Create symbol-date index
        df['date_idx'] = pd.Categorical(df['timestamp']).codes
        df['scaled_idx'] = range(len(df))

        # Process each day
        for day_idx, current_date in enumerate(dates):
            if day_idx < sequence_length:
                continue

            if day_idx % 500 == 0:
                print(f"Day {day_idx}/{len(dates)}: {current_date}, Equity: ₹{capital:,.0f}")

            day_df = df[df['timestamp'] == current_date]

            # Update existing positions
            for symbol in list(positions.keys()):
                position = positions[symbol]

                # Calculate days held (calendar days)
                days_held = (pd.Timestamp(current_date) - pd.Timestamp(position['entry_date'])).days

                symbol_row = day_df[day_df['symbol'] == symbol]

                # Force exit if no data and held too long (max 10 calendar days = ~7 trading days)
                if symbol_row.empty:
                    if days_held >= 10:  # Force exit stale positions
                        # Use last known price
                        last_price = position.get('last_price', position['entry_price'])
                        profit_pct = (last_price - position['entry_price']) / position['entry_price']
                        profit = profit_pct * position['invested']
                        capital += position['invested'] + profit

                        self.trades.append({
                            'trade_id': position['trade_id'],
                            'symbol': symbol,
                            'entry_date': position['entry_date'],
                            'exit_date': current_date,
                            'entry_price': position['entry_price'],
                            'exit_price': last_price,
                            'quantity': position['quantity'],
                            'profit_pct': profit_pct,
                            'profit_amount': profit,
                            'exit_reason': 'FORCE_EXIT_NO_DATA',
                            'days_held': days_held,
                            'confidence': position['confidence'],
                            'volatility': position['volatility'],
                            'position_multiplier': 1.0,
                            'partial_exits': 0
                        })
                        del positions[symbol]
                    continue

                current_price = symbol_row['close'].values[0]
                position['last_price'] = current_price  # Track last known price

                # Update trade monitor
                exit_signal = trade_monitor.update_trade(
                    position['trade_id'],
                    current_price,
                    current_date
                )

                if exit_signal:
                    # Close position
                    profit = exit_signal['total_profit_amount']
                    capital += position['invested'] + profit

                    self.trades.append({
                        'trade_id': exit_signal['trade_id'],
                        'symbol': symbol,
                        'entry_date': position['entry_date'],
                        'exit_date': current_date,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'quantity': position['quantity'],
                        'profit_pct': exit_signal['profit_pct'],
                        'profit_amount': profit,
                        'exit_reason': exit_signal['exit_reason'],
                        'days_held': exit_signal['days_held'],
                        'confidence': position['confidence'],
                        'volatility': position['volatility'],
                        'position_multiplier': exit_signal.get('position_multiplier', 1.0),
                        'partial_exits': exit_signal.get('partial_exits_done', 0)
                    })

                    del positions[symbol]

            # Look for new entries
            if len(positions) < max_positions:
                # Get symbols not in position and allowed
                available = day_df[
                    (~day_df['symbol'].isin(positions.keys())) &
                    (~day_df['symbol'].isin(self.config.excluded_stocks))
                ]

                if len(available) > 0:
                    # Prepare sequences for batch prediction
                    sequences = []
                    symbols_to_eval = []
                    symbol_data = []

                    for _, row in available.iterrows():
                        symbol = row['symbol']
                        # Get historical data for this symbol
                        symbol_hist = df[(df['symbol'] == symbol) &
                                        (df['timestamp'] <= current_date)].tail(sequence_length)

                        if len(symbol_hist) < sequence_length:
                            continue

                        # Get scaled features
                        hist_indices = symbol_hist['scaled_idx'].values
                        seq_features = df_scaled[hist_indices]
                        sequences.append(seq_features)
                        symbols_to_eval.append(symbol)
                        symbol_data.append(row)

                    if sequences:
                        # Batch predict (GPU accelerated)
                        batch = np.array(sequences)
                        predictions, confidences = self.predict_batch(batch)

                        # Find bullish signals
                        for i, (symbol, row) in enumerate(zip(symbols_to_eval, symbol_data)):
                            pred = predictions[i]
                            conf = confidences[i]

                            # Only enter if prediction is bullish (>0.5)
                            if pred > 0.5 and len(positions) < max_positions:
                                # Calculate volatility
                                symbol_df = df[df['symbol'] == symbol]
                                idx = symbol_df[symbol_df['timestamp'] == current_date].index[0]
                                local_idx = symbol_df.index.get_loc(idx)
                                volatility = self._calculate_volatility(symbol_df.reset_index(), local_idx)
                                atr_pct = self._calculate_atr_pct(symbol_df.reset_index(), local_idx)

                                # Position sizing (V3: confidence + volatility based)
                                position_mult = self.strategy.calculate_position_size_multiplier(conf, volatility)
                                base_position = capital / max_positions
                                position_value = base_position * position_mult

                                if position_value < 1000:  # Minimum position
                                    continue

                                entry_price = row['close']
                                quantity = int(position_value / entry_price)

                                if quantity > 0:
                                    trade_id = f"T{trade_id_counter:05d}"
                                    trade_id_counter += 1

                                    # Add to monitor
                                    trade_monitor.add_trade(
                                        trade_id=trade_id,
                                        symbol=symbol,
                                        entry_price=entry_price,
                                        entry_date=current_date,
                                        quantity=quantity,
                                        confidence=conf,
                                        volatility=volatility,
                                        atr_pct=atr_pct
                                    )

                                    # Track position
                                    invested = quantity * entry_price
                                    capital -= invested

                                    positions[symbol] = {
                                        'trade_id': trade_id,
                                        'entry_date': current_date,
                                        'entry_price': entry_price,
                                        'quantity': quantity,
                                        'invested': invested,
                                        'confidence': conf,
                                        'volatility': volatility
                                    }

            # Record equity
            position_value = sum(
                p['quantity'] * df[(df['symbol'] == sym) &
                                  (df['timestamp'] == current_date)]['close'].values[0]
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
                    'trade_id': position['trade_id'],
                    'symbol': symbol,
                    'entry_date': position['entry_date'],
                    'exit_date': final_date,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'quantity': position['quantity'],
                    'profit_pct': profit_pct,
                    'profit_amount': profit,
                    'exit_reason': 'END_OF_BACKTEST',
                    'days_held': (pd.Timestamp(final_date) - pd.Timestamp(position['entry_date'])).days,
                    'confidence': position['confidence'],
                    'volatility': position['volatility'],
                    'position_multiplier': 1.0,
                    'partial_exits': 0
                })

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics"""
        if not self.trades:
            return {}

        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)

        # Basic stats
        total_trades = len(trades_df)
        winners = trades_df[trades_df['profit_pct'] > 0]
        losers = trades_df[trades_df['profit_pct'] <= 0]

        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        # Confidence breakdown
        high_conf = trades_df[trades_df['confidence'] >= 0.8]
        med_conf = trades_df[(trades_df['confidence'] >= 0.6) & (trades_df['confidence'] < 0.8)]
        low_conf = trades_df[trades_df['confidence'] < 0.6]

        # Returns
        total_profit = trades_df['profit_amount'].sum()
        avg_return = trades_df['profit_pct'].mean()
        total_return = (equity_df['total_equity'].iloc[-1] - self.initial_capital) / self.initial_capital

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

        # CAGR
        years = (pd.Timestamp(equity_df['date'].iloc[-1]) - pd.Timestamp(equity_df['date'].iloc[0])).days / 365.25
        if years > 0:
            cagr = ((equity_df['total_equity'].iloc[-1] / self.initial_capital) ** (1/years)) - 1
        else:
            cagr = 0

        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'win_rate_high_conf': len(high_conf[high_conf['profit_pct'] > 0]) / len(high_conf) if len(high_conf) > 0 else 0,
            'win_rate_med_conf': len(med_conf[med_conf['profit_pct'] > 0]) / len(med_conf) if len(med_conf) > 0 else 0,
            'win_rate_low_conf': len(low_conf[low_conf['profit_pct'] > 0]) / len(low_conf) if len(low_conf) > 0 else 0,
            'trades_high_conf': len(high_conf),
            'trades_med_conf': len(med_conf),
            'trades_low_conf': len(low_conf),
            'avg_return': avg_return,
            'total_return': total_return,
            'cagr': cagr,
            'total_profit': total_profit,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'avg_hold_days': trades_df['days_held'].mean(),
            'final_equity': equity_df['total_equity'].iloc[-1]
        }

        return metrics

    def print_results(self, metrics: Dict):
        """Print formatted results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS V3 - FULLY OPTIMIZED")
        print("="*60)

        print(f"\nTotal Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
        print(f"  - High Confidence ({metrics['trades_high_conf']} trades): {metrics['win_rate_high_conf']*100:.1f}%")
        print(f"  - Medium Confidence ({metrics['trades_med_conf']} trades): {metrics['win_rate_med_conf']*100:.1f}%")
        print(f"  - Low Confidence ({metrics['trades_low_conf']} trades): {metrics['win_rate_low_conf']*100:.1f}%")

        print(f"\nAverage Return: {metrics['avg_return']*100:.2f}%")
        print(f"Total Return: {metrics['total_return']*100:.2f}%")
        print(f"CAGR: {metrics['cagr']*100:.2f}%")

        print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")

        print(f"\nAvg Hold Days: {metrics['avg_hold_days']:.1f}")
        print(f"Final Equity: ₹{metrics['final_equity']:,.0f}")

    def save_results(self, output_dir: str):
        """Save results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv(output_path / 'all_trades_v3.csv', index=False)

        # Save equity curve
        if self.equity_curve:
            equity_df = pd.DataFrame(self.equity_curve)
            equity_df.to_csv(output_path / 'equity_curve_v3.csv', index=False)

        # Exit reason analysis
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            exit_analysis = trades_df.groupby('exit_reason').agg({
                'trade_id': 'count',
                'profit_pct': ['mean', lambda x: (x > 0).mean()],
                'profit_amount': 'sum'
            }).round(4)
            exit_analysis.columns = ['trades', 'avg_return', 'win_rate', 'total_profit']
            exit_analysis = exit_analysis.sort_values('total_profit', ascending=False)
            exit_analysis.to_csv(output_path / 'exit_reason_analysis_v3.csv')

        print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Swing Backtester V3')
    parser.add_argument('--model', type=str, nargs='+', required=True,
                       help='Model path(s) - multiple for ensemble')
    parser.add_argument('--scaler', type=str, required=True)
    parser.add_argument('--features', type=str, required=True)
    parser.add_argument('--capital', type=float, default=100000)
    parser.add_argument('--output', type=str, default='reports/swing_backtest_v3/')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--temperature', type=float, default=0.5,
                       help='Temperature for confidence calibration (0.3-1.0)')
    parser.add_argument('--max-positions', type=int, default=10)

    args = parser.parse_args()

    # Device selection
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Run backtest
    backtester = SwingBacktesterV3(
        model_paths=args.model,
        scaler_path=args.scaler,
        feature_dir=args.features,
        initial_capital=args.capital,
        device=device,
        temperature=args.temperature
    )

    metrics = backtester.run_backtest(max_positions=args.max_positions)
    backtester.print_results(metrics)
    backtester.save_results(args.output)


if __name__ == '__main__':
    main()
