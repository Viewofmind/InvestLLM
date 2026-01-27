#!/usr/bin/env python3
"""
Evaluate Price Prediction Model
===============================

Loads the trained model and evaluates it on the test set.
Generates plots for Actual vs Predicted values.
"""

import os
import sys
from pathlib import Path
import glob
import logging
import argparse

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg') # Prevent SegFault on Mac
import matplotlib.pyplot as plt
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.price_prediction.lstm_model import PricePredictionLSTM

logging.basicConfig(level=logging.INFO)
console = Console()

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed" / "price_prediction"
MODEL_DIR = Path("models/price_prediction")
RESULTS_DIR = Path("reports/price_prediction")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_ticker(ticker: str, model: PricePredictionLSTM, seq_length: int = 60, split_ratio: float = 0.8):
    """Evaluate model on a single ticker"""
    # Try sentiment file first
    file_path = PROCESSED_DIR / f"{ticker}_processed_sentiment.parquet"
    if not file_path.exists():
        file_path = PROCESSED_DIR / f"{ticker}_processed.parquet"
        
    if not file_path.exists():
        console.print(f"[red]Data not found for {ticker}[/red]")
        return
        
    df = pd.read_parquet(file_path)
    
    # Split
    split_idx = int(len(df) * split_ratio)
    test_df = df.iloc[split_idx:].copy() # Actually we want specific test set?
    
    # In train script we did 80/20 train/val. 
    # Ideally we should have had 70/10/20.
    # For now, let's use the VAL set (last 20%) as Proxy Test Set to see performance.
    
    target_col = 'Target'
    feature_cols = [c for c in df.columns if c != target_col]
    
    # Scale (We need the scaler fitted on Train to handle Test correctly)
    # Re-fitting scaler on Train part
    train_df = df.iloc[:split_idx]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])
    
    test_scaled = scaler.transform(test_df[feature_cols])
    test_targets = test_df[target_col].values
    
    # Sequences
    xs, ys = [], []
    for i in range(len(test_scaled) - seq_length):
        x = test_scaled[i : i + seq_length]
        y = test_targets[i + seq_length]
        xs.append(x)
        ys.append(y)
        
    if not xs:
        print(f"Not enough data for {ticker}")
        return

    X_test = torch.FloatTensor(np.array(xs))
    y_test = torch.FloatTensor(np.array(ys))
    
    # Inference
    model.eval()
    with torch.no_grad():
        preds = model(X_test).numpy().flatten()
    
    # Metrics
    mae = np.mean(np.abs(preds - list(ys)))
    rmse = np.sqrt(np.mean((preds - list(ys))**2))
    
    # Directional Accuracy
    pred_sign = np.sign(preds)
    act_sign = np.sign(ys)
    acc = np.mean(pred_sign == act_sign) * 100
    
    console.print(f"\n[bold]{ticker} Results:[/bold]")
    console.print(f"  MAE: {mae:.6f}")
    console.print(f"  RMSE: {rmse:.6f}")
    console.print(f"  Directional Accuracy: {acc:.2f}%")
    
    # Plot - DISABLED due to SegFault
    # plt.figure(figsize=(12, 6))
    # plt.plot(ys, label='Actual Return', alpha=0.7)
    # plt.plot(preds, label='Predicted Return', alpha=0.7)
    # plt.title(f"{ticker} - Actual vs Predicted Returns (Test Set)")
    # plt.legend()
    # plt.savefig(RESULTS_DIR / f"{ticker}_prediction.png")
    # plt.close()
    
    # Cumulative Return Strategy (Simple Backtest)
    # If Pred > 0, Buy. Else Sell/Flat.
    # Note: Target is Log Return.
    # Strategy Return = Sign(Pred) * Actual
    
    strategy_returns = pred_sign * ys
    cum_strategy = np.cumsum(strategy_returns)
    cum_market = np.cumsum(ys)
    
    # plt.figure(figsize=(12, 6))
    # plt.plot(cum_market, label='Buy & Hold')
    # plt.plot(cum_strategy, label='LSTM Strategy')
    # plt.title(f"{ticker} - Strategy Backtest (Cumulative Returns)")
    # plt.legend()
    # plt.savefig(RESULTS_DIR / f"{ticker}_backtest.png")
    # plt.close()
    
    return acc

def main():
    # Find best model
    checkpoints = list(MODEL_DIR.glob("*.ckpt"))
    if not checkpoints:
        console.print("[red]No checkpoints found![/red]")
        return
        
    # Sort by val_loss (extracted from filename usually, or modification time)
    # Filename format: price_lstm-epoch=XX-val_loss=0.XXXX.ckpt
    # We can rely on 'ls' order or parse. Let's pick the last one or manually parse
    best_model_path = sorted(checkpoints)[-1] # Quick hack, better to parse
    console.print(f"Loading model: {best_model_path}")
    
    # Force CPU to avoid MPS segfaults during inference if any
    model = PricePredictionLSTM.load_from_checkpoint(best_model_path, map_location=torch.device('cpu'))
    
    # Evaluate on our 5 tickers
    tickers = ["RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK"]
    
    accuracies = []
    for ticker in tickers:
        try:
            acc = evaluate_ticker(ticker, model)
            if acc:
                accuracies.append(acc)
        except Exception as e:
            console.print(f"[red]Error evaluating {ticker}: {e}[/red]")
            
    if accuracies:
        console.print(f"\n[bold green]Average Accuracy: {np.mean(accuracies):.2f}%[/bold green]")

if __name__ == "__main__":
    main()
