#!/usr/bin/env python3
"""
InvestLLM Ensemble Backtester - RunPod GPU
==========================================
Full backtest with:
- Price Model (LSTM)
- Sentiment Model (FinBERT)
- Fundamental Scorer
- Smart Exit Risk Management
- Meta-Learner combining all signals
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/processed/price_prediction")
ENSEMBLE_MODEL_DIR = Path("models/ensemble_trained")
SENTIMENT_MODEL_DIR = Path("models/sentiment/sentiment_model_final")
SEQ_LENGTH = 60
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# TRADING DECISION TYPES
# ============================================================================

class TradingDecision(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"

class ExitReason(Enum):
    PROFIT_TARGET_50 = "Profit Target 50%"
    PROFIT_TARGET_100 = "Profit Target 100%"
    PROFIT_TARGET_200 = "Profit Target 200%"
    MA_EXIT = "MA Trend Reversal"
    MODEL_EXIT = "Model Bearish"
    STOP_LOSS = "Stop Loss"
    CATASTROPHIC = "Catastrophic Loss"
    TIME_EXIT = "Time-based Exit"
    SIGNAL_FLIP = "Signal Flip"
    END_OF_DATA = "End of Data"

# ============================================================================
# SMART EXIT MANAGER
# ============================================================================

class SmartExitManager:
    """Intelligent exit strategy with partial profit taking"""

    def __init__(self):
        self.positions = {}

        # Profit targets (partial exits)
        self.profit_targets = [
            (0.50, 0.25),   # At 50% profit, exit 25%
            (1.00, 0.25),   # At 100% profit, exit 25%
            (2.00, 0.25),   # At 200% profit, exit 25%
        ]

        # Risk parameters
        self.stop_loss = -0.15           # 15% stop loss
        self.catastrophic_loss = -0.50   # 50% emergency exit
        self.ma_period = 50              # MA for trend
        self.max_hold_days = 252         # 1 year max hold

    def register_position(self, ticker: str, entry_price: float, entry_date, entry_pred: float):
        self.positions[ticker] = {
            'entry_price': entry_price,
            'entry_date': entry_date,
            'entry_pred': entry_pred,
            'targets_hit': set(),
            'highest_price': entry_price
        }

    def check_exit(self, ticker: str, current_price: float, prices: pd.Series,
                   current_pred: float, current_date) -> Tuple[bool, float, ExitReason]:
        """
        Check if position should be exited.
        Returns: (should_exit, exit_portion, reason)
        """
        if ticker not in self.positions:
            return False, 0.0, ExitReason.END_OF_DATA

        pos = self.positions[ticker]
        entry_price = pos['entry_price']
        pnl = (current_price - entry_price) / entry_price

        # Update highest price
        pos['highest_price'] = max(pos['highest_price'], current_price)

        # 1. Catastrophic loss protection
        if pnl <= self.catastrophic_loss:
            return True, 1.0, ExitReason.CATASTROPHIC

        # 2. Stop loss
        if pnl <= self.stop_loss:
            return True, 1.0, ExitReason.STOP_LOSS

        # 3. Profit targets (partial exits)
        for target_pct, exit_pct in self.profit_targets:
            if pnl >= target_pct and target_pct not in pos['targets_hit']:
                pos['targets_hit'].add(target_pct)
                reason = ExitReason.PROFIT_TARGET_50 if target_pct == 0.5 else \
                         ExitReason.PROFIT_TARGET_100 if target_pct == 1.0 else \
                         ExitReason.PROFIT_TARGET_200
                return True, exit_pct, reason

        # 4. MA trend reversal (if we have enough data)
        if len(prices) >= self.ma_period:
            ma = prices.iloc[-self.ma_period:].mean()
            if current_price < ma * 0.98 and pnl > 0:  # Below MA with profit
                return True, 0.5, ExitReason.MA_EXIT

        # 5. Model turns bearish
        if current_pred < -0.001 and pnl > 0.1:  # Model bearish, have 10%+ profit
            return True, 0.5, ExitReason.MODEL_EXIT

        # 6. Time-based exit
        if hasattr(current_date, 'days'):
            days_held = (current_date - pos['entry_date']).days
        else:
            days_held = 0
        if days_held > self.max_hold_days:
            return True, 1.0, ExitReason.TIME_EXIT

        return False, 0.0, ExitReason.END_OF_DATA

    def clear_position(self, ticker: str):
        if ticker in self.positions:
            del self.positions[ticker]

# ============================================================================
# META LEARNER
# ============================================================================

class MetaLearner:
    """Combines signals from multiple models"""

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'price': 0.50,
            'sentiment': 0.20,
            'fundamental': 0.30
        }

        self.thresholds = {
            'strong_buy': 0.6,
            'buy': 0.2,
            'sell': -0.2,
            'strong_sell': -0.6
        }

    def predict(self, price_signal: float, sentiment_signal: float,
                fundamental_signal: float) -> Tuple[TradingDecision, float, float]:
        """
        Combine signals and return decision.
        Returns: (decision, ensemble_signal, confidence)
        """
        # Normalize fundamental to -1 to +1
        fund_normalized = (fundamental_signal - 0.5) * 2

        # Weighted combination
        ensemble = (
            price_signal * self.weights['price'] +
            sentiment_signal * self.weights['sentiment'] +
            fund_normalized * self.weights['fundamental']
        )

        # Calculate confidence based on signal agreement
        signals = [price_signal, sentiment_signal, fund_normalized]
        signs = [np.sign(s) for s in signals if abs(s) > 0.1]

        if len(signs) >= 2:
            agreement = abs(sum(signs)) / len(signs)
            confidence = 0.5 + 0.5 * agreement
        else:
            confidence = 0.5

        # Make decision
        if ensemble >= self.thresholds['strong_buy']:
            decision = TradingDecision.STRONG_BUY
        elif ensemble >= self.thresholds['buy']:
            decision = TradingDecision.BUY
        elif ensemble <= self.thresholds['strong_sell']:
            decision = TradingDecision.STRONG_SELL
        elif ensemble <= self.thresholds['sell']:
            decision = TradingDecision.SELL
        else:
            decision = TradingDecision.HOLD

        return decision, ensemble, confidence

# ============================================================================
# FUNDAMENTAL SCORER
# ============================================================================

class FundamentalScorer:
    """Simple rule-based fundamental scoring"""

    def score(self, ticker: str) -> float:
        """Return fundamental score 0-1 (placeholder - returns neutral)"""
        # In production, would fetch real fundamental data
        return 0.5  # Neutral

# ============================================================================
# LSTM MODEL LOADER
# ============================================================================

def load_lstm_model(model_dir: Path):
    """Load trained LSTM model"""
    import pytorch_lightning as pl

    # Find best checkpoint
    checkpoints = list(model_dir.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {model_dir}")

    # Get best model (lowest val_loss in filename)
    best_ckpt = min(checkpoints, key=lambda x: float(str(x).split('val_loss=')[1].split('-')[0])
                    if 'val_loss=' in str(x) else float('inf'))

    print(f"Loading model: {best_ckpt.name}")

    # Load checkpoint
    checkpoint = torch.load(best_ckpt, map_location=DEVICE)

    # Get hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    input_size = hparams.get('input_size', 22)
    hidden_size = hparams.get('hidden_size', 256)
    num_layers = hparams.get('num_layers', 3)
    dropout = hparams.get('dropout', 0.3)

    # Recreate model architecture
    class LSTMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            self.attention = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_size, 1)
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(hidden_size * 2, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, hidden_size // 2),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size // 2, 1)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
            return self.fc(context).squeeze(-1)

    model = LSTMModel()

    # Load state dict (handle Lightning wrapper)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '').replace('network.', '')
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

    return model, input_size

# ============================================================================
# BACKTESTER
# ============================================================================

def backtest_ticker(ticker_file: Path, lstm_model, input_size: int,
                    meta_learner: MetaLearner, exit_manager: SmartExitManager,
                    fundamental_scorer: FundamentalScorer, scaler: StandardScaler) -> Optional[Dict]:
    """Backtest single ticker with full ensemble"""

    try:
        df = pd.read_parquet(ticker_file)
        ticker = ticker_file.stem.split('_')[0]

        if len(df) < SEQ_LENGTH + 200:
            return None

        # Prepare features
        features = [c for c in df.columns if c not in ['Date', 'Target']]

        # Add sentiment placeholder column if needed
        feature_data = df[features].values
        if feature_data.shape[1] < input_size:
            padding = np.zeros((len(df), input_size - feature_data.shape[1]))
            feature_data = np.hstack([feature_data, padding])

        # Scale
        feature_scaled = scaler.transform(feature_data)

        targets = df['Target'].values
        closes = df['Close'].values
        dates = df.index

        # Split
        split_idx = int(len(df) * 0.8)
        test_indices = [i for i in range(split_idx, len(df)) if i >= SEQ_LENGTH]

        if not test_indices:
            return None

        # Get fundamental score (constant for ticker)
        fund_score = fundamental_scorer.score(ticker)

        # Trading simulation
        trades = []
        position = 0.0
        entry_price = 0.0
        entry_date = None
        total_realized_pnl = 0.0

        for idx in test_indices:
            current_date = dates[idx]
            current_price = closes[idx]

            # Get LSTM prediction
            seq = feature_scaled[idx-SEQ_LENGTH:idx]
            X = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                price_signal = lstm_model(X).cpu().numpy()[0]
            price_signal = np.clip(price_signal * 100, -1, 1)

            # Sentiment signal (placeholder - would use real news)
            sentiment_signal = 0.0

            # Get ensemble decision
            decision, ensemble_signal, confidence = meta_learner.predict(
                price_signal, sentiment_signal, fund_score
            )

            is_last = (idx == test_indices[-1])

            # Check exits
            if position > 0:
                should_exit, exit_portion, exit_reason = exit_manager.check_exit(
                    ticker, current_price, pd.Series(closes[:idx+1]),
                    price_signal, current_date
                )

                # Also exit on sell signals
                if decision in [TradingDecision.STRONG_SELL, TradingDecision.SELL]:
                    should_exit = True
                    exit_portion = 1.0
                    exit_reason = ExitReason.SIGNAL_FLIP

                if is_last:
                    should_exit = True
                    exit_portion = 1.0
                    exit_reason = ExitReason.END_OF_DATA

                if should_exit and position > 0:
                    pnl = (current_price - entry_price) / entry_price
                    exit_size = min(exit_portion, position)
                    realized_pnl = pnl * exit_size
                    total_realized_pnl += realized_pnl
                    position -= exit_size

                    if position <= 0.01:
                        days_held = (current_date - entry_date).days if hasattr(current_date - entry_date, 'days') else 0
                        trades.append({
                            'Ticker': ticker,
                            'Entry Date': entry_date,
                            'Entry Price': entry_price,
                            'Exit Date': current_date,
                            'Exit Price': current_price,
                            'PnL': pnl,
                            'Realized PnL': total_realized_pnl,
                            'Exit Reason': exit_reason.value,
                            'Days Held': days_held,
                            'Confidence': confidence
                        })

                        position = 0.0
                        total_realized_pnl = 0.0
                        exit_manager.clear_position(ticker)

            # New entry
            if position == 0 and not is_last:
                if decision in [TradingDecision.STRONG_BUY, TradingDecision.BUY]:
                    if confidence >= 0.5:
                        position = 1.0
                        entry_price = current_price
                        entry_date = current_date
                        exit_manager.register_position(ticker, entry_price, entry_date, price_signal)

        if not trades:
            return None

        # Calculate metrics
        trade_df = pd.DataFrame(trades)
        total_pnl = trade_df['Realized PnL'].sum()
        win_trades = trade_df[trade_df['PnL'] > 0]
        win_rate = len(win_trades) / len(trade_df) * 100

        # Time period
        start_date = pd.to_datetime(dates[test_indices[0]])
        end_date = pd.to_datetime(dates[test_indices[-1]])
        days = (end_date - start_date).days
        years = days / 365.25

        # CAGR
        final_ratio = 1 + total_pnl
        cagr = (final_ratio ** (1/years) - 1) * 100 if years > 0 and final_ratio > 0 else 0

        # Sharpe ratio
        if len(trade_df) > 1:
            returns = trade_df['PnL'].values
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                trades_per_year = len(trade_df) / years if years > 0 else len(trade_df)
                sharpe = (avg_return / std_return) * np.sqrt(trades_per_year)
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        cumulative = (1 + trade_df['PnL']).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_dd = drawdown.min() * 100

        return {
            'Ticker': ticker,
            'Total Return %': total_pnl * 100,
            'CAGR %': cagr,
            'Sharpe': sharpe,
            'Win Rate %': win_rate,
            'Max Drawdown %': max_dd,
            'Trades': len(trade_df),
            'Avg Days Held': trade_df['Days Held'].mean(),
            'Years': years
        }

    except Exception as e:
        print(f"Error {ticker_file.stem}: {e}")
        return None

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("InvestLLM ENSEMBLE BACKTESTER - GPU")
    print("="*70)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load LSTM model
    print("Loading LSTM model...")
    lstm_model, input_size = load_lstm_model(ENSEMBLE_MODEL_DIR)
    print(f"Input size: {input_size}")

    # Load scaler
    import pickle
    scaler_path = ENSEMBLE_MODEL_DIR / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Loaded scaler")
    else:
        print("Warning: No scaler found, fitting new one")
        scaler = StandardScaler()
        # Fit on first file
        first_file = list(DATA_DIR.glob("*_processed.parquet"))[0]
        df = pd.read_parquet(first_file)
        features = [c for c in df.columns if c not in ['Date', 'Target']]
        feature_data = df[features].values
        if feature_data.shape[1] < input_size:
            padding = np.zeros((len(df), input_size - feature_data.shape[1]))
            feature_data = np.hstack([feature_data, padding])
        scaler.fit(feature_data)

    # Initialize components
    meta_learner = MetaLearner(weights={'price': 0.50, 'sentiment': 0.20, 'fundamental': 0.30})
    exit_manager = SmartExitManager()
    fundamental_scorer = FundamentalScorer()

    # Get stock files
    files = list(DATA_DIR.glob("*_processed.parquet"))
    print(f"\nBacktesting {len(files)} stocks...")
    print()

    # Run backtest
    results = []
    from tqdm import tqdm

    for f in tqdm(files, desc="Backtesting"):
        exit_manager = SmartExitManager()  # Fresh for each ticker
        res = backtest_ticker(f, lstm_model, input_size, meta_learner,
                              exit_manager, fundamental_scorer, scaler)
        if res:
            results.append(res)

    # Aggregate results
    df_res = pd.DataFrame(results)

    if df_res.empty:
        print("No results!")
        return

    # Save detailed results
    df_res.to_csv("ensemble_backtest_results.csv", index=False)
    print(f"\nSaved detailed results to ensemble_backtest_results.csv")

    # Print summary
    print("\n" + "="*70)
    print("ENSEMBLE BACKTEST RESULTS")
    print("="*70)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-"*40)
    print(f"{'Stocks Tested':<25} {len(df_res):>15}")
    print(f"{'Avg Test Period':<25} {df_res['Years'].mean():>14.1f}y")
    print(f"{'Avg Total Return':<25} {df_res['Total Return %'].mean():>14.2f}%")
    print(f"{'Avg CAGR':<25} {df_res['CAGR %'].mean():>14.2f}%")
    print(f"{'Avg Sharpe Ratio':<25} {df_res['Sharpe'].mean():>15.2f}")
    print(f"{'Avg Win Rate':<25} {df_res['Win Rate %'].mean():>14.1f}%")
    print(f"{'Avg Max Drawdown':<25} {df_res['Max Drawdown %'].mean():>14.1f}%")
    print(f"{'Avg Trades/Stock':<25} {df_res['Trades'].mean():>15.1f}")
    print(f"{'Avg Days Held':<25} {df_res['Avg Days Held'].mean():>15.0f}")

    # Top performers
    print("\n" + "-"*70)
    print("TOP 10 PERFORMERS")
    print("-"*70)
    top10 = df_res.nlargest(10, 'Total Return %')
    for _, row in top10.iterrows():
        print(f"{row['Ticker']:<12} Return: {row['Total Return %']:>8.1f}%  "
              f"CAGR: {row['CAGR %']:>6.1f}%  Sharpe: {row['Sharpe']:>5.2f}  "
              f"Win: {row['Win Rate %']:>5.1f}%")

    # Bottom performers
    print("\n" + "-"*70)
    print("BOTTOM 5 PERFORMERS")
    print("-"*70)
    bottom5 = df_res.nsmallest(5, 'Total Return %')
    for _, row in bottom5.iterrows():
        print(f"{row['Ticker']:<12} Return: {row['Total Return %']:>8.1f}%  "
              f"CAGR: {row['CAGR %']:>6.1f}%  Sharpe: {row['Sharpe']:>5.2f}  "
              f"Win: {row['Win Rate %']:>5.1f}%")

    # Risk metrics
    print("\n" + "-"*70)
    print("RISK ANALYSIS")
    print("-"*70)
    print(f"Profitable Stocks: {len(df_res[df_res['Total Return %'] > 0])} / {len(df_res)} "
          f"({len(df_res[df_res['Total Return %'] > 0])/len(df_res)*100:.1f}%)")
    print(f"Avg Sharpe (profitable): {df_res[df_res['Total Return %'] > 0]['Sharpe'].mean():.2f}")
    print(f"Worst Drawdown: {df_res['Max Drawdown %'].min():.1f}%")

    print("\n" + "="*70)
    print("BACKTEST COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
