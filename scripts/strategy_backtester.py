#!/usr/bin/env python3
"""
Strategy Backtester
===================
Simulates a trading strategy based on the trained LSTM Price Prediction model.
Aggregates results across all tickers to estimate Portfolio Performance.
"""

import os
import sys
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.price_prediction.lstm_model import PricePredictionLSTM

# Configuration
# Force CPU for inference (simpler, avoids device mismatch if trained on GPU)
DEVICE = torch.device('cpu') 
PROCESSED_DIR = Path("data/processed/price_prediction")
MODELS_DIR = Path("models/price_prediction")

console = Console()

def load_best_model():
    checkpoints = list(MODELS_DIR.glob("*.ckpt"))
    if not checkpoints:
        raise FileNotFoundError("No model checkpoints found!")
    # Sort by modification time to get latest
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

def backtest_ticker(ticker_file, model, seq_length=60):
    try:
        df = pd.read_parquet(ticker_file)
        if len(df) < seq_length + 100:
            return None # Not enough data
            
        # We only backtest on the "Test" portion (last 20% by default split)
        split_idx = int(len(df) * 0.8)
        test_df = df.iloc[split_idx:].copy()
        
        # Prepare Sequences
        # (Simplified: In a real backtest we'd step day-by-day. Here we batch predict for speed)
        # We need the features X. Assuming X columns are everything except 'Target' and 'Date'
        # Note: The 'Target' is Next Day Return.
        # We need to reconstruct the input sequences.
        # Ideally we should use same Dataset class, but let's do quick construction:
        
        # FIX: Model was trained on ALL features including OHLCV.
        features = [c for c in df.columns if c not in ['Date', 'Target']]

        # FIX: Apply StandardScaler - fit on train data only, transform all
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_data = df.iloc[:split_idx][features].values
        scaler.fit(train_data)  # Fit only on training portion
        feature_data = scaler.transform(df[features].values)  # Transform all data

        targets = df['Target'].values
        
        
        test_indices = range(split_idx, len(df))
        X_test = []
        y_test = []
        test_dates = []
        test_closes = []
        
        for i in test_indices:
            if i < seq_length: continue
            seq = feature_data[i-seq_length:i]
            target = targets[i]
            X_test.append(seq)
            y_test.append(target)
            test_dates.append(df.index[i])
            test_closes.append(df['Close'].iloc[i])
            
        if not X_test:
            return None
            
        X_tensor = torch.FloatTensor(np.array(X_test)).to(DEVICE)
        y_true = np.array(y_test)
        
        # Predict
        with torch.no_grad():
            preds = model(X_tensor).numpy().flatten()
            
        # Strategy Logic
        # If Pred > 0.0005 (0.05%), Buy.
        # If Pred < -0.0005, Short.
        # Else Flat.
        threshold = 0.0005
        signals = np.zeros_like(preds)
        signals[preds > threshold] = 1  # Long
        signals[preds < -threshold] = -1 # Short
        
        
        # --- Trade Logging Logic ---
        trades = []
        position = 0
        entry_price = 0.0
        entry_idx = 0
        entry_reason = ""
        
        # Use aligned lists
        dates = test_dates
        closes = test_closes
        
        for i, signal in enumerate(signals):
            current_date = dates[i]
            current_price = closes[i]
            current_pred = preds[i]
            
            # Check for close/flip or End of Data
            is_last_step = (i == len(signals) - 1)
            
            if position != 0:
                if signal != position or is_last_step: # Signal changed or flat or End of Data -> Exit
                    # Determine Status
                    # It is Open only if we are at the last step AND the signal hasn't flipped
                    status = "Open" if (is_last_step and signal == position) else "Closed"
                    
                    # Exit Trade (Mark to Market for Open)
                    exit_price = current_price
                    
                    # Calculate Stats
                    if position == 1: # Long
                        pnl = (exit_price - entry_price) / entry_price
                        # Max Drawdown during hold (using Close prices)
                        trade_prices = closes[entry_idx : i+1]
                        min_price = np.min(trade_prices)
                        max_dd = (min_price - entry_price) / entry_price
                        direction = "Long"
                    else: # Short
                        pnl = (entry_price - exit_price) / entry_price
                        # Max Drawdown for Short (Price going UP is bad)
                        trade_prices = closes[entry_idx : i+1]
                        max_price = np.max(trade_prices)
                        max_dd = (entry_price - max_price) / entry_price
                        direction = "Short"

                    trades.append({
                        "Ticker": ticker_file.stem.split('_')[0],
                        "Entry Date": dates[entry_idx],
                        "Entry Price": entry_price,
                        "Exit Date": current_date,
                        "Exit Price": exit_price,
                        "Direction": direction,
                        "Status": status, # New Column
                        "PnL": pnl,
                        "Max DD %": max_dd * 100,
                        "Reason": entry_reason,
                        "Days Held": (i - entry_idx)
                    })
                    position = 0
            
            # Check for new entry (Only if not last step)
            if position == 0 and signal != 0 and not is_last_step:
                position = signal
                entry_price = current_price
                entry_idx = i
                entry_reason = f"Model Pred: {current_pred:.5f}"
                
        # --- End Trade Logging ---
        
        # Strategy Daily Returns
        strat_returns = signals * y_true
        
        # Metrics
        total_return = np.sum(strat_returns)
        win_rate = np.mean(np.sign(strat_returns) == 1) if np.sum(signals!=0) > 0 else 0
        
        if np.std(strat_returns) == 0:
            sharpe = 0
        else:
            sharpe = (np.mean(strat_returns) / np.std(strat_returns)) * np.sqrt(252)
            
        cum_ret = np.exp(np.cumsum(strat_returns))
        max_dd = calculate_max_drawdown(cum_ret)
        
        # CAGR Calculation
        start_date = pd.to_datetime(dates[0])
        end_date = pd.to_datetime(dates[-1])
        days = (end_date - start_date).days
        years = days / 365.25
        
        if years > 0:
            # Using simple return sum approximation for CAGR base or proper cumulative product?
            # Since we used log returns (strat_returns approx log returns), 
            # Cumulative Return Factor = exp(sum(log_returns))
            # Final Value = Initial * exp(total_return)  <-- Validation check?
            # total_return variable above is sum(signals * y_true). y_true is log return.
            # So sum is Total Log Return.
            # Final Ratio = exp(total_return)
            
            final_ratio = np.exp(total_return)
            cagr = (final_ratio ** (1 / years)) - 1
        else:
            cagr = 0.0
            
        return {
            "Summary": {
                "Ticker": ticker_file.stem.split('_')[0],
                "Total Return": total_return * 100, # This is Log Return sum? NO. 
                # Wait. In previous steps `total_return` was just printed as sum?
                # If y_true are log returns, sum is Total Log Return.
                # If we convert to % for display, we should strictly use (exp(sum) - 1) * 100.
                # But for small values sum approx works. For 50%, exp(0.51) - 1 = 66%.
                # Let's fix Total Return to be accurate arithmetic return too.
                "Total Return %": (np.exp(total_return) - 1) * 100, 
                "CAGR %": cagr * 100,
                "Years": years,
                "Sharpe": sharpe,
                "Max Drawdown": max_dd * 100,
                "Win Rate": win_rate * 100,
                "Trades": np.sum(signals != 0) 
            },
            "Trades": trades
        }
        
    except Exception as e:
        console.print(f"[red]Error {ticker_file.stem}: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None

def main():
    console.print("[bold green]Starting Portfolio Backtest...[/bold green]")
    model = load_best_model()
    
    # Prioritize sentiment files
    files = list(PROCESSED_DIR.glob("*_sentiment.parquet"))
    if not files:
        files = list(PROCESSED_DIR.glob("*_processed.parquet"))
        
    console.print(f"Backtesting on {len(files)} stocks...")
    
    results = []
    all_trades = []
    
    from rich.progress import track
    for f in track(files, description="Simulating trading..."):
        res = backtest_ticker(f, model)
        if res:
            results.append(res['Summary'])
            all_trades.extend(res['Trades'])
            
    # Save Trade Log
    if all_trades:
        trade_df = pd.DataFrame(all_trades)
        trade_df.to_csv("trades_log.csv", index=False)
        console.print(f"[bold yellow]Saved {len(trade_df)} trades to trades_log.csv[/bold yellow]")
            
    # Aggregation
    df_res = pd.DataFrame(results)
    
    # Portfolio Metrics (Equal weighted average of all strategies)
    avg_return = df_res['Total Return %'].mean()
    avg_cagr = df_res['CAGR %'].mean()
    avg_sharpe = df_res['Sharpe'].mean()
    avg_dd = df_res['Max Drawdown'].mean()
    avg_win = df_res['Win Rate'].mean()
    avg_years = df_res['Years'].mean()
    
    # Display Table
    table = Table(title="Strategy Performance Summary (Test Set)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Stocks Tested", str(len(df_res)))
    table.add_row("Avg Years Tested", f"{avg_years:.1f} Years")
    table.add_row("Avg Total Return", f"{avg_return:.2f}%")
    table.add_row("Avg CAGR", f"{avg_cagr:.2f}%")
    table.add_row("Avg Annual Sharpe", f"{avg_sharpe:.2f}")
    table.add_row("Avg Max Drawdown", f"{avg_dd:.2f}%")
    table.add_row("Avg Win Rate", f"{avg_win:.2f}%")
    
    console.print(table)
    
    # Top Performers
    console.print("\n[bold]Top 5 Performers:[/bold]")
    top_5 = df_res.nlargest(5, 'Total Return %')
    
    top_table = Table()
    top_table.add_column("Ticker")
    top_table.add_column("Return", style="green")
    top_table.add_column("CAGR", style="blue")
    top_table.add_column("Sharpe")
    for _, row in top_5.iterrows():
        top_table.add_row(
            row['Ticker'], 
            f"{row['Total Return %']:.2f}%", 
            f"{row['CAGR %']:.2f}%",
            f"{row['Sharpe']:.2f}"
        )
        
    console.print(top_table)
    console.print(f"\n[dim]Note: Backtest period approx {avg_years:.1f} years.[/dim]")

if __name__ == "__main__":
    main()
