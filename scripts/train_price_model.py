#!/usr/bin/env python3
"""
Train Price Prediction Model
============================

Trains an LSTM model on the processed market data.

Usage:
    python scripts/train_price_model.py
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from rich.console import Console

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.price_prediction.lstm_model import PricePredictionLSTM

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed" / "price_prediction"
MODEL_DIR = Path("models/price_prediction")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class MarketDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_col: str = 'Target'):
        self.data = torch.FloatTensor(data.drop(columns=[target_col]).values).to(torch.float32)
        self.targets = torch.FloatTensor(data[target_col].values).to(torch.float32)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def prepare_data(batch_size: int = 32, seq_length: int = 60, split_ratio: float = 0.8):
    """
    Load and prepare data for training.
    
    IMPORTANT: We must do a time-series split, not random shuffle.
    Since we have multiple tickers, we split each ticker's data individually.
    """
    # Look for sentiment augmented files first
    files = list(PROCESSED_DIR.glob("*_sentiment.parquet"))
    if not files:
        # Fallback to base processed files
        files = list(PROCESSED_DIR.glob("*_processed.parquet"))
        
    if not files:
        raise ValueError(f"No processed data found in {PROCESSED_DIR}")
        
    console.print(f"Found {len(files)} ticker files (Sentiment Augmented: {'_sentiment' in str(files[0])}).")
    
    train_datasets = []
    val_datasets = []
    
    total_samples = 0
    input_size = 0
    
    for file in files:
        df = pd.read_parquet(file)
        
        # Determine sequence length from data shape (reconstruct from prepare_market_data logic)
        # Note: prepare_market_data saved (N, Features+Target)
        # It's not (N, Seq, Features) yet?
        # WAIT. `prepare_market_data.py` created sequences (N, Seq, Feat)? 
        # checking `prepare_market_data.py`:
        # It calls `create_sequences` but then returns `data` which is `df[available_features + ['Target']].values`?
        # NO. `prepare_market_data.py` did NOT actually save sequences. 
        # It saved the raw derived features dataframe.
        # We need to create sequences HERE or fix `prepare_market_data.py`.
        # `prepare_market_data.py` has a `create_sequences` method but it wasn't used in `process_ticker` main flow!
        # It just returned the DF.
        # So we need to create sequences here.
        
        # Split index
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        
        # Scale (Fit on TRAIN only)
        # Exclude Target from scaling? Usually better to scale features only.
        # But for LSTM stability, scaling target is also good if it's large.
        # Log Returns are small (-0.1 to 0.1), so maybe okay without scaling, 
        # but let's standard scale features.
        
        target_col = 'Target'
        feature_cols = [c for c in df.columns if c != target_col]
        input_size = len(feature_cols)
        
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])
        
        # Create sequences
        def create_seq(data_scaled, targets, seq_len):
            xs, ys = [], []
            for i in range(len(data_scaled) - seq_len):
                x = data_scaled[i : i + seq_len]
                y = targets[i + seq_len]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)
            
        train_targets = train_df[target_col].values
        val_targets = val_df[target_col].values
        
        X_train, y_train = create_seq(train_scaled, train_targets, seq_length)
        X_val, y_val = create_seq(val_scaled, val_targets, seq_length)
        
        if len(X_train) > 0:
            train_datasets.append(torch.utils.data.TensorDataset(
                torch.FloatTensor(X_train), torch.FloatTensor(y_train)
            ))
        if len(X_val) > 0:
            val_datasets.append(torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val), torch.FloatTensor(y_val)
            ))
            
    if not train_datasets:
        raise ValueError("Not enough data to create sequences.")
        
    train_ds = ConcatDataset(train_datasets)
    val_ds = ConcatDataset(val_datasets)
    
    console.print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4) # Shuffle train batches is OK
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, input_size


def main(batch_size: int = 64, epochs: int = 20, lr: float = 1e-3):
    pl.seed_everything(42)
    
    # 1. Prepared Data
    console.print("[bold]Preparing Data...[/bold]")
    train_loader, val_loader, input_size = prepare_data(batch_size=batch_size)
    
    # 2. Init Model
    console.print(f"[bold]Initializing Model (Input Size: {input_size})...[/bold]")
    model = PricePredictionLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        learning_rate=lr
    )
    
    # 3. Setup Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_DIR,
        filename='price_lstm-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )
    
    # Detect best accelerator
    import torch
    if torch.cuda.is_available():
        accelerator = "cuda"
        console.print("[bold cyan]Using CUDA GPU for training![/bold cyan]")
    else:
        accelerator = "cpu"
        console.print("[yellow]Using CPU for training[/yellow]")
        # Note: MPS (Mac Metal) requires newer PyTorch Lightning version

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop],
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=50,  # Reduce logging overhead
        enable_progress_bar=True
    )
    
    # 4. Train
    console.print("[bold green]Starting Training...[/bold green]")
    trainer.fit(model, train_loader, val_loader)
    
    console.print(f"Best model path: {checkpoint_callback.best_model_path}")
    
    # Save metrics?
    # TODO: Load best model and eval


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
