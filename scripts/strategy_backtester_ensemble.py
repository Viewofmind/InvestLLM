#!/usr/bin/env python3
"""
Ensemble Strategy Backtester
=============================
Combines three AI models for trading decisions:
1. Price Model (LSTM) - Technical/momentum signal
2. Sentiment Model (FinBERT) - News sentiment signal
3. Fundamental Model (XGBoost) - Value/quality signal

Uses MetaLearner to combine signals into final trading decisions.
Includes Smart Exit risk management.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.price_prediction.lstm_model import PricePredictionLSTM
from investllm.strategies.smart_exit_manager import SmartExitManager, RECOMMENDED_CONFIG
from investllm.models.ensemble.meta_learner import MetaLearner, TradingDecision
from investllm.models.fundamental.quality_scorer import FundamentalScorer

# Configuration
DEVICE = torch.device('cpu')
PROCESSED_DIR = Path("data/processed/price_prediction")
MODELS_DIR = Path("models/price_prediction")
SENTIMENT_MODEL_DIR = Path("models/sentiment/sentiment_model_final")
FUNDAMENTAL_DATA_DIR = Path("data/processed/fundamentals")

console = Console()


@dataclass
class ModelSignals:
    """Container for all model signals"""
    price_signal: float      # -1 to +1
    sentiment_signal: float  # -1 to +1
    fundamental_signal: float  # 0 to 1


class EnsembleBacktester:
    """
    Backtest engine using ensemble of AI models.
    """

    def __init__(
        self,
        price_model_path: Optional[str] = None,
        sentiment_model_path: Optional[str] = None,
        use_fundamental: bool = True,
        meta_strategy: str = "weighted_average",
        meta_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ensemble backtester.

        Args:
            price_model_path: Path to LSTM checkpoint
            sentiment_model_path: Path to FinBERT model
            use_fundamental: Whether to use fundamental scoring
            meta_strategy: MetaLearner strategy
            meta_weights: Custom weights for meta-learner
        """
        self.price_model = None
        self.sentiment_model = None
        self.fundamental_scorer = None
        self.meta_learner = None
        self.scaler = StandardScaler()

        # Load price model
        self._load_price_model(price_model_path)

        # Load sentiment model (if available)
        self._load_sentiment_model(sentiment_model_path)

        # Initialize fundamental scorer
        if use_fundamental:
            self.fundamental_scorer = FundamentalScorer()

        # Initialize meta-learner
        self.meta_learner = MetaLearner(
            strategy=meta_strategy,
            weights=meta_weights
        )

        # Smart exit manager
        self.exit_manager = SmartExitManager(RECOMMENDED_CONFIG)

    def _load_price_model(self, model_path: Optional[str] = None):
        """Load LSTM price prediction model"""
        if model_path:
            path = Path(model_path)
        else:
            # Find latest checkpoint
            checkpoints = list(MODELS_DIR.glob("*.ckpt"))
            if not checkpoints:
                console.print("[yellow]Warning: No price model found. Using price signal = 0[/yellow]")
                return
            path = max(checkpoints, key=os.path.getmtime)

        console.print(f"[blue]Loading Price Model:[/blue] {path.name}")
        self.price_model = PricePredictionLSTM.load_from_checkpoint(path, map_location=DEVICE)
        self.price_model.eval()
        self.price_model.freeze()

    def _load_sentiment_model(self, model_path: Optional[str] = None):
        """Load FinBERT sentiment model"""
        # Check for sentiment model
        if model_path:
            path = Path(model_path)
        else:
            path = SENTIMENT_MODEL_DIR  # Already points to sentiment_model_final

        if path.exists() and (path / "config.json").exists():
            try:
                # Import sentiment scorer
                from investllm.models.sentiment.sentiment_scorer import SentimentScorer
                self.sentiment_model = SentimentScorer(str(path))
                console.print(f"[blue]Loading Sentiment Model:[/blue] {path.name}")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load sentiment model: {e}[/yellow]")
                self.sentiment_model = None
        else:
            console.print("[yellow]Warning: No sentiment model found. Using sentiment signal = 0[/yellow]")

    def get_price_signal(self, feature_data: np.ndarray, seq_length: int = 60) -> float:
        """
        Get signal from LSTM price model.

        Args:
            feature_data: Scaled feature array (seq_length, n_features)
            seq_length: Sequence length

        Returns:
            Signal from -1 to +1
        """
        if self.price_model is None:
            return 0.0

        if len(feature_data) < seq_length:
            return 0.0

        # Get last sequence
        seq = feature_data[-seq_length:]
        X = torch.FloatTensor(seq).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = self.price_model(X).numpy().flatten()[0]

        # Convert prediction to signal (-1 to +1)
        # Model predicts expected return, scale to signal
        signal = np.clip(pred * 100, -1, 1)  # Scale and clip

        return float(signal)

    def get_sentiment_signal(self, ticker: str, date: pd.Timestamp) -> float:
        """
        Get signal from sentiment model.

        For now, returns neutral if no sentiment model/data.
        In production, would fetch news and analyze.

        Args:
            ticker: Stock ticker
            date: Current date

        Returns:
            Signal from -1 to +1
        """
        if self.sentiment_model is None:
            return 0.0

        # TODO: Implement actual sentiment scoring
        # Would fetch news for ticker around date and score
        # For now, return neutral
        return 0.0

    def get_fundamental_signal(self, ticker: str, fundamentals: Dict) -> float:
        """
        Get signal from fundamental scorer.

        Args:
            ticker: Stock ticker
            fundamentals: Dictionary of fundamental metrics

        Returns:
            Score from 0 to 1
        """
        if self.fundamental_scorer is None:
            return 0.5  # Neutral

        if not fundamentals:
            return 0.5

        score = self.fundamental_scorer.score(ticker, fundamentals)
        return score.composite_score

    def backtest_ticker(
        self,
        ticker_file: Path,
        seq_length: int = 60,
        fundamentals: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        Backtest a single ticker using ensemble model.

        Args:
            ticker_file: Path to parquet file with price data
            seq_length: LSTM sequence length
            fundamentals: Fundamental data for this ticker

        Returns:
            Dictionary with backtest results
        """
        try:
            df = pd.read_parquet(ticker_file)
            ticker = ticker_file.stem.split('_')[0]

            if len(df) < seq_length + 200:
                return None

            # Split data (80% train, 20% test)
            split_idx = int(len(df) * 0.8)

            # Prepare features
            features = [c for c in df.columns if c not in ['Date', 'Target']]

            # Scale features (fit on train only)
            train_data = df.iloc[:split_idx][features].values
            self.scaler.fit(train_data)
            feature_data = self.scaler.transform(df[features].values)

            targets = df['Target'].values
            closes = df['Close'].values
            dates = df.index

            # Get fundamental signal (constant for ticker)
            fund_signal = self.get_fundamental_signal(ticker, fundamentals or {})

            # Test period indices
            test_indices = [i for i in range(split_idx, len(df)) if i >= seq_length]

            if not test_indices:
                return None

            # Trading simulation
            trades = []
            position = 0.0
            entry_price = 0.0
            entry_date = None
            predictions_history = []
            total_realized_pnl = 0.0
            partial_exits = []

            for idx in test_indices:
                current_date = dates[idx]
                current_price = closes[idx]

                # Get ensemble signals
                price_signal = self.get_price_signal(feature_data[:idx+1], seq_length)
                sent_signal = self.get_sentiment_signal(ticker, current_date)

                # Get ensemble prediction
                ensemble = self.meta_learner.predict(
                    ticker=ticker,
                    date=str(current_date),
                    price_signal=price_signal,
                    sentiment_signal=sent_signal,
                    fundamental_signal=fund_signal
                )

                predictions_history.append(price_signal)
                is_last_step = (idx == test_indices[-1])

                # Trading logic based on ensemble decision
                if position > 0:
                    # Check smart exit
                    price_series = pd.Series(closes[:idx+1])
                    exit_signal = self.exit_manager.check_exit(
                        ticker=ticker,
                        prices=price_series,
                        predictions=predictions_history[-10:] if len(predictions_history) >= 10 else predictions_history,
                        current_date=current_date
                    )

                    should_exit = exit_signal.should_exit
                    exit_portion = exit_signal.exit_portion
                    exit_reason = exit_signal.reason.value

                    # Also exit on strong sell or end of data
                    if ensemble.decision in [TradingDecision.STRONG_SELL, TradingDecision.SELL]:
                        should_exit = True
                        exit_portion = 1.0
                        exit_reason = f"Ensemble: {ensemble.decision.value}"

                    if is_last_step:
                        should_exit = True
                        exit_portion = 1.0
                        exit_reason = "End of Data"

                    if should_exit and position > 0:
                        pnl = (current_price - entry_price) / entry_price
                        exit_size = min(exit_portion, position)
                        realized_pnl = pnl * exit_size
                        total_realized_pnl += realized_pnl

                        partial_exits.append({
                            "Date": current_date,
                            "Price": current_price,
                            "Portion": exit_size,
                            "PnL": pnl,
                            "Reason": exit_reason
                        })

                        position -= exit_size

                        if position <= 0.01:
                            days_held = (current_date - entry_date).days if entry_date else 0

                            trades.append({
                                "Ticker": ticker,
                                "Entry Date": entry_date,
                                "Entry Price": entry_price,
                                "Exit Date": current_date,
                                "Exit Price": current_price,
                                "Direction": "Long",
                                "PnL": (current_price - entry_price) / entry_price,
                                "Realized PnL": total_realized_pnl,
                                "Exit Reason": exit_reason,
                                "Partial Exits": len(partial_exits),
                                "Days Held": days_held,
                                "Ensemble Signal": ensemble.ensemble_signal,
                                "Confidence": ensemble.confidence
                            })

                            position = 0.0
                            total_realized_pnl = 0.0
                            partial_exits = []
                            predictions_history = []

                # New entry on BUY/STRONG_BUY with sufficient confidence
                if position == 0 and not is_last_step:
                    if ensemble.decision in [TradingDecision.STRONG_BUY, TradingDecision.BUY]:
                        if ensemble.confidence >= 0.5:  # Minimum confidence
                            position = ensemble.position_size  # Use suggested size
                            entry_price = current_price
                            entry_date = current_date
                            predictions_history = [price_signal]

                            self.exit_manager.register_position(
                                ticker=ticker,
                                entry_price=entry_price,
                                entry_date=entry_date,
                                entry_prediction=price_signal
                            )

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
            cagr = (final_ratio ** (1 / years) - 1) * 100 if years > 0 and final_ratio > 0 else 0

            # Sharpe
            if len(trade_df) > 1:
                returns = trade_df['PnL'].values
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 / (days / len(returns))) if np.std(returns) > 0 else 0
            else:
                sharpe = 0

            # Average confidence
            avg_confidence = trade_df['Confidence'].mean() if 'Confidence' in trade_df.columns else 0

            return {
                "Summary": {
                    "Ticker": ticker,
                    "Total Return %": total_pnl * 100,
                    "CAGR %": cagr,
                    "Years": years,
                    "Sharpe": sharpe,
                    "Win Rate": win_rate,
                    "Trades": len(trade_df),
                    "Avg Days Held": trade_df['Days Held'].mean(),
                    "Avg Confidence": avg_confidence,
                    "Fundamental Score": fund_signal
                },
                "Trades": trades
            }

        except Exception as e:
            console.print(f"[red]Error {ticker_file.stem}: {e}[/red]")
            import traceback
            traceback.print_exc()
            return None


def load_fundamentals() -> Dict[str, Dict]:
    """
    Load fundamental data for all tickers.

    Returns:
        Dictionary mapping ticker to fundamental metrics
    """
    fundamentals = {}

    # Try to load from fundamentals directory
    fund_file = FUNDAMENTAL_DATA_DIR / "nifty100_fundamentals.parquet"

    if fund_file.exists():
        df = pd.read_parquet(fund_file)
        for _, row in df.iterrows():
            ticker = row.get('symbol', row.get('ticker', ''))
            fundamentals[ticker] = row.to_dict()
    else:
        console.print("[yellow]No fundamental data found. Using neutral scores.[/yellow]")

    return fundamentals


def main():
    console.print("[bold green]╔════════════════════════════════════════════════════╗[/bold green]")
    console.print("[bold green]║     ENSEMBLE STRATEGY BACKTESTER                   ║[/bold green]")
    console.print("[bold green]║     Price + Sentiment + Fundamentals               ║[/bold green]")
    console.print("[bold green]╚════════════════════════════════════════════════════╝[/bold green]")
    console.print()

    # Initialize backtester
    backtester = EnsembleBacktester(
        use_fundamental=True,
        meta_strategy="weighted_average",
        meta_weights={
            'price': 0.50,      # Higher weight since sentiment not trained yet
            'sentiment': 0.10,  # Low weight until model is ready
            'fundamental': 0.40
        }
    )

    # Load fundamental data
    fundamentals = load_fundamentals()

    # Get processed files
    files = list(PROCESSED_DIR.glob("*_processed.parquet"))
    console.print(f"[cyan]Backtesting {len(files)} stocks with ensemble model...[/cyan]")
    console.print()

    # Show model weights
    console.print("[dim]Model Weights:[/dim]")
    console.print(f"  [blue]Price (LSTM):[/blue]      {backtester.meta_learner.weights['price']:.0%}")
    console.print(f"  [green]Sentiment:[/green]        {backtester.meta_learner.weights['sentiment']:.0%}")
    console.print(f"  [yellow]Fundamental:[/yellow]      {backtester.meta_learner.weights['fundamental']:.0%}")
    console.print()

    results = []
    all_trades = []

    from rich.progress import track
    for f in track(files, description="Running Ensemble Backtest..."):
        ticker = f.stem.split('_')[0]
        fund_data = fundamentals.get(ticker, {})

        res = backtester.backtest_ticker(f, fundamentals=fund_data)
        if res:
            results.append(res['Summary'])
            all_trades.extend(res['Trades'])

    # Save trades
    if all_trades:
        trade_df = pd.DataFrame(all_trades)
        trade_df.to_csv("trades_ensemble.csv", index=False)
        console.print(f"[bold yellow]Saved {len(trade_df)} trades to trades_ensemble.csv[/bold yellow]")

    # Aggregate results
    df_res = pd.DataFrame(results)

    if df_res.empty:
        console.print("[red]No results generated![/red]")
        return

    # Portfolio metrics
    avg_return = df_res['Total Return %'].mean()
    avg_cagr = df_res['CAGR %'].mean()
    avg_sharpe = df_res['Sharpe'].mean()
    avg_win = df_res['Win Rate'].mean()
    avg_years = df_res['Years'].mean()
    avg_confidence = df_res['Avg Confidence'].mean()

    # Display results
    console.print()
    table = Table(title="Ensemble Strategy Performance (Test Set)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Stocks Tested", str(len(df_res)))
    table.add_row("Test Period", f"{avg_years:.1f} Years")
    table.add_row("Avg Total Return", f"{avg_return:.2f}%")
    table.add_row("Avg CAGR", f"{avg_cagr:.2f}%")
    table.add_row("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
    table.add_row("Avg Win Rate", f"{avg_win:.1f}%")
    table.add_row("Avg Confidence", f"{avg_confidence:.2f}")

    console.print(table)

    # Top performers
    console.print("\n[bold]Top 5 Performers:[/bold]")
    top_5 = df_res.nlargest(5, 'Total Return %')

    top_table = Table()
    top_table.add_column("Ticker")
    top_table.add_column("Return", style="green")
    top_table.add_column("CAGR", style="blue")
    top_table.add_column("Sharpe")
    top_table.add_column("Win Rate")
    top_table.add_column("Fund Score")

    for _, row in top_5.iterrows():
        top_table.add_row(
            row['Ticker'],
            f"{row['Total Return %']:.1f}%",
            f"{row['CAGR %']:.1f}%",
            f"{row['Sharpe']:.2f}",
            f"{row['Win Rate']:.0f}%",
            f"{row['Fundamental Score']:.2f}"
        )

    console.print(top_table)

    # Bottom performers
    console.print("\n[bold]Bottom 5 Performers:[/bold]")
    bottom_5 = df_res.nsmallest(5, 'Total Return %')

    bottom_table = Table()
    bottom_table.add_column("Ticker")
    bottom_table.add_column("Return", style="red")
    bottom_table.add_column("CAGR", style="blue")
    bottom_table.add_column("Win Rate")

    for _, row in bottom_5.iterrows():
        bottom_table.add_row(
            row['Ticker'],
            f"{row['Total Return %']:.1f}%",
            f"{row['CAGR %']:.1f}%",
            f"{row['Win Rate']:.0f}%"
        )

    console.print(bottom_table)

    # Summary
    console.print()
    console.print("[bold]Model Components:[/bold]")
    console.print(f"  ✓ Price Model (LSTM): {'[green]Loaded[/green]' if backtester.price_model else '[red]Not Found[/red]'}")
    console.print(f"  {'✓' if backtester.sentiment_model else '○'} Sentiment Model: {'[green]Loaded[/green]' if backtester.sentiment_model else '[yellow]Not Trained Yet[/yellow]'}")
    console.print(f"  ✓ Fundamental Scorer: [green]Active[/green]")
    console.print()
    console.print("[dim]Note: Once sentiment model is trained, update weights to balance all three signals.[/dim]")


if __name__ == "__main__":
    main()
