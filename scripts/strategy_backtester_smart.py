#!/usr/bin/env python3
"""
Strategy Backtester with Smart Exit Risk Management
====================================================
Integrates SmartExitManager for intelligent exit strategies:
- Partial profit taking at 50%, 100%, 200%, 300% targets
- MA-based trend reversal exits
- Model-based exits when LSTM turns bearish
- Momentum exits (RSI-based)
- Catastrophic loss protection (>50% drop)
- Time-based exit for dead money
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
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.price_prediction.lstm_model import PricePredictionLSTM
from investllm.strategies.smart_exit_manager import SmartExitManager, RECOMMENDED_CONFIG, ExitReason

# Configuration
DEVICE = torch.device('cpu')
PROCESSED_DIR = Path("data/processed/price_prediction")
MODELS_DIR = Path("models/price_prediction")

console = Console()


def load_best_model():
    """Load the best model checkpoint"""
    checkpoints = list(MODELS_DIR.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError("No model checkpoints found!")
    latest_model = max(checkpoints, key=os.path.getmtime)
    console.print(f"[blue]Loading Model:[/blue] {latest_model.name}")
    model = PricePredictionLSTM.load_from_checkpoint(latest_model, map_location=DEVICE)
    model.eval()
    model.freeze()
    return model


def calculate_max_drawdown(cumulative_returns):
    """Calculate the Maximum Drawdown of a cumulative return series."""
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def backtest_ticker_smart(ticker_file, model, seq_length=60, exit_config=None):
    """
    Backtest a single ticker with Smart Exit Risk Management.

    Key differences from basic backtester:
    - Uses SmartExitManager for intelligent exits
    - Supports partial position exits at profit targets
    - Protects gains with MA-based exits
    - Emergency exit on catastrophic drops
    """
    try:
        df = pd.read_parquet(ticker_file)
        ticker = ticker_file.stem.split('_')[0]

        if len(df) < seq_length + 200:  # Need more data for 200 MA
            return None

        # Split data
        split_idx = int(len(df) * 0.8)

        # Prepare features
        features = [c for c in df.columns if c not in ['Date', 'Target']]

        # Scale features
        scaler = StandardScaler()
        train_data = df.iloc[:split_idx][features].values
        scaler.fit(train_data)
        feature_data = scaler.transform(df[features].values)

        targets = df['Target'].values
        closes = df['Close'].values
        dates = df.index

        # Generate predictions for test period
        test_indices = list(range(split_idx, len(df)))
        X_test = []

        for i in test_indices:
            if i < seq_length:
                continue
            seq = feature_data[i-seq_length:i]
            X_test.append(seq)

        if not X_test:
            return None

        # Batch predict
        X_tensor = torch.FloatTensor(np.array(X_test)).to(DEVICE)
        with torch.no_grad():
            preds = model(X_tensor).numpy().flatten()

        # Align predictions with test indices
        valid_test_indices = [i for i in test_indices if i >= seq_length]
        pred_map = {idx: pred for idx, pred in zip(valid_test_indices, preds)}

        # Initialize SmartExitManager
        config = exit_config or RECOMMENDED_CONFIG
        exit_manager = SmartExitManager(config)

        # Trading simulation with Smart Exit
        trades = []
        position = 0.0  # Now tracks fractional position (0.0 to 1.0)
        entry_price = 0.0
        entry_idx = 0
        entry_date = None
        entry_pred = 0.0
        predictions_history = []

        # Track partial exits
        total_realized_pnl = 0.0
        partial_exits = []

        threshold = 0.0005

        for idx in valid_test_indices:
            current_date = dates[idx]
            current_price = closes[idx]
            current_pred = pred_map.get(idx, 0)
            predictions_history.append(current_pred)

            # Determine signal
            signal = 0
            if current_pred > threshold:
                signal = 1  # Long
            elif current_pred < -threshold:
                signal = -1  # Short

            is_last_step = (idx == valid_test_indices[-1])

            # If we have a position, check Smart Exit
            if position > 0:
                # Get price history for exit calculations
                price_series = pd.Series(closes[:idx+1])

                # Check smart exit
                exit_signal = exit_manager.check_exit(
                    ticker=ticker,
                    prices=price_series,
                    predictions=predictions_history[-10:] if len(predictions_history) >= 10 else predictions_history,
                    current_date=current_date
                )

                should_exit = exit_signal.should_exit
                exit_portion = exit_signal.exit_portion
                exit_reason = exit_signal.reason.value

                # Force exit on last step or signal flip to short
                if is_last_step or signal == -1:
                    should_exit = True
                    exit_portion = 1.0  # Exit remaining
                    exit_reason = "End of Data" if is_last_step else "Signal Flip to Short"

                if should_exit and position > 0:
                    # Calculate PnL for this exit
                    pnl = (current_price - entry_price) / entry_price
                    exit_size = min(exit_portion, position)
                    realized_pnl = pnl * exit_size
                    total_realized_pnl += realized_pnl

                    # Track partial exit
                    partial_exits.append({
                        "Date": current_date,
                        "Price": current_price,
                        "Portion": exit_size,
                        "PnL": pnl,
                        "Reason": exit_reason
                    })

                    # Update position
                    position -= exit_size

                    # If fully exited, record trade
                    if position <= 0.01:  # Essentially zero
                        days_held = (current_date - entry_date).days if entry_date else 0

                        trades.append({
                            "Ticker": ticker,
                            "Entry Date": entry_date,
                            "Entry Price": entry_price,
                            "Exit Date": current_date,
                            "Exit Price": current_price,
                            "Direction": "Long",
                            "Status": "Open" if is_last_step else "Closed",
                            "PnL": (current_price - entry_price) / entry_price,
                            "Realized PnL": total_realized_pnl,
                            "Max DD %": 0,  # Calculated below
                            "Exit Reason": exit_reason,
                            "Partial Exits": len(partial_exits),
                            "Days Held": days_held
                        })

                        # Reset
                        position = 0.0
                        total_realized_pnl = 0.0
                        partial_exits = []
                        predictions_history = []

            # New entry (only Long for now, avoiding Short complexity)
            if position == 0 and signal == 1 and not is_last_step:
                position = 1.0
                entry_price = current_price
                entry_idx = idx
                entry_date = current_date
                entry_pred = current_pred
                predictions_history = [current_pred]

                # Register with exit manager
                exit_manager.register_position(
                    ticker=ticker,
                    entry_price=entry_price,
                    entry_date=entry_date,
                    entry_prediction=entry_pred
                )

        if not trades:
            return None

        # Calculate metrics
        trade_df = pd.DataFrame(trades)
        total_pnl = trade_df['Realized PnL'].sum() if 'Realized PnL' in trade_df.columns else trade_df['PnL'].sum()
        win_trades = trade_df[trade_df['PnL'] > 0]
        win_rate = len(win_trades) / len(trade_df) * 100 if len(trade_df) > 0 else 0

        # Time period
        start_date = pd.to_datetime(dates[valid_test_indices[0]])
        end_date = pd.to_datetime(dates[valid_test_indices[-1]])
        days = (end_date - start_date).days
        years = days / 365.25

        # CAGR
        final_ratio = 1 + total_pnl
        cagr = (final_ratio ** (1 / years) - 1) * 100 if years > 0 else 0

        # Sharpe (simplified)
        if len(trade_df) > 1:
            returns = trade_df['PnL'].values
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252 / (days / len(returns))) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Exit reason breakdown
        exit_reasons = trade_df['Exit Reason'].value_counts().to_dict() if 'Exit Reason' in trade_df.columns else {}

        return {
            "Summary": {
                "Ticker": ticker,
                "Total Return %": total_pnl * 100,
                "CAGR %": cagr,
                "Years": years,
                "Sharpe": sharpe,
                "Max Drawdown": -30,  # Placeholder
                "Win Rate": win_rate,
                "Trades": len(trade_df),
                "Avg Days Held": trade_df['Days Held'].mean() if 'Days Held' in trade_df.columns else 0
            },
            "Trades": trades,
            "Exit Reasons": exit_reasons
        }

    except Exception as e:
        console.print(f"[red]Error {ticker_file.stem}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def main():
    console.print("[bold green]Starting Portfolio Backtest with SMART EXIT...[/bold green]")
    console.print("[yellow]Risk Management: Partial Targets + MA Exit + Catastrophic Protection[/yellow]\n")

    model = load_best_model()

    # Use processed files only (no sentiment)
    files = list(PROCESSED_DIR.glob("*_processed.parquet"))
    console.print(f"Backtesting on {len(files)} stocks...")

    results = []
    all_trades = []
    all_exit_reasons = {}

    from rich.progress import track
    for f in track(files, description="Simulating with Smart Exit..."):
        res = backtest_ticker_smart(f, model)
        if res:
            results.append(res['Summary'])
            all_trades.extend(res['Trades'])
            # Aggregate exit reasons
            for reason, count in res.get('Exit Reasons', {}).items():
                all_exit_reasons[reason] = all_exit_reasons.get(reason, 0) + count

    # Save Trade Log
    if all_trades:
        trade_df = pd.DataFrame(all_trades)
        trade_df.to_csv("trades_smart_exit.csv", index=False)
        console.print(f"[bold yellow]Saved {len(trade_df)} trades to trades_smart_exit.csv[/bold yellow]")

    # Aggregation
    df_res = pd.DataFrame(results)

    if df_res.empty:
        console.print("[red]No results generated![/red]")
        return

    # Portfolio Metrics
    avg_return = df_res['Total Return %'].mean()
    avg_cagr = df_res['CAGR %'].mean()
    avg_sharpe = df_res['Sharpe'].mean()
    avg_win = df_res['Win Rate'].mean()
    avg_years = df_res['Years'].mean()
    avg_days_held = df_res['Avg Days Held'].mean()

    # Display Table
    table = Table(title="Smart Exit Strategy Performance (Test Set)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Stocks Tested", str(len(df_res)))
    table.add_row("Avg Years Tested", f"{avg_years:.1f} Years")
    table.add_row("Avg Total Return", f"{avg_return:.2f}%")
    table.add_row("Avg CAGR", f"{avg_cagr:.2f}%")
    table.add_row("Avg Annual Sharpe", f"{avg_sharpe:.2f}")
    table.add_row("Avg Win Rate", f"{avg_win:.2f}%")
    table.add_row("Avg Days Held", f"{avg_days_held:.0f} days")

    console.print(table)

    # Exit Reasons Summary
    if all_exit_reasons:
        console.print("\n[bold]Exit Reasons Breakdown:[/bold]")
        reason_table = Table()
        reason_table.add_column("Exit Reason", style="yellow")
        reason_table.add_column("Count", style="white")
        reason_table.add_column("%", style="green")

        total_exits = sum(all_exit_reasons.values())
        for reason, count in sorted(all_exit_reasons.items(), key=lambda x: -x[1]):
            pct = count / total_exits * 100
            reason_table.add_row(reason, str(count), f"{pct:.1f}%")

        console.print(reason_table)

    # Top Performers
    console.print("\n[bold]Top 5 Performers:[/bold]")
    top_5 = df_res.nlargest(5, 'Total Return %')

    top_table = Table()
    top_table.add_column("Ticker")
    top_table.add_column("Return", style="green")
    top_table.add_column("CAGR", style="blue")
    top_table.add_column("Win Rate")
    top_table.add_column("Trades")

    for _, row in top_5.iterrows():
        top_table.add_row(
            row['Ticker'],
            f"{row['Total Return %']:.2f}%",
            f"{row['CAGR %']:.2f}%",
            f"{row['Win Rate']:.1f}%",
            str(int(row['Trades']))
        )

    console.print(top_table)

    # Bottom Performers
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
            f"{row['Total Return %']:.2f}%",
            f"{row['CAGR %']:.2f}%",
            f"{row['Win Rate']:.1f}%"
        )

    console.print(bottom_table)

    console.print(f"\n[dim]Smart Exit Config: Targets at 50%, 100%, 200%, 300% | MA Exit: 50 EMA | Catastrophic: 50%[/dim]")


if __name__ == "__main__":
    main()
