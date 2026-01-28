#!/usr/bin/env python3
"""
InvestLLM Cloud Training Script for RunPod
==========================================
Self-contained training script for GPU training on RunPod.

Usage:
    python train_runpod.py --epochs 50 --batch_size 256

Requirements:
    pip install torch pytorch-lightning pandas numpy scikit-learn pyarrow rich
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================================
# LSTM MODEL DEFINITION
# ============================================================================

class PricePredictionLSTM(pl.LightningModule):
    """LSTM model for price prediction"""

    def __init__(self, input_size=21, hidden_size=128, num_layers=2,
                 dropout=0.2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.loss_fn = nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = self.loss_fn(y_hat, y)

        # Calculate directional accuracy
        pred_direction = (y_hat > 0).float()
        true_direction = (y > 0).float()
        accuracy = (pred_direction == true_direction).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }


# ============================================================================
# DATASET CLASS
# ============================================================================

class PriceDataset(Dataset):
    """Dataset for price prediction sequences"""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


# ============================================================================
# DATA PREPARATION
# ============================================================================

def create_sequences(data_scaled, targets, seq_len):
    """Create sequences for LSTM training"""
    xs, ys = [], []
    for i in range(len(data_scaled) - seq_len):
        x = data_scaled[i : i + seq_len]
        y = targets[i + seq_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


def prepare_data(data_dir, seq_length=60, split_ratio=0.8, batch_size=128):
    """Load and prepare data for training"""

    data_path = Path(data_dir)
    files = list(data_path.glob("*_processed.parquet"))

    if not files:
        raise FileNotFoundError(f"No processed parquet files found in {data_dir}")

    console.print(f"[green]Found {len(files)} ticker files[/green]")

    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []

    for file_path in files:
        df = pd.read_parquet(file_path)

        if len(df) < seq_length + 100:
            continue

        # Get features and target
        feature_cols = [c for c in df.columns if c not in ['Date', 'Target']]

        # Time-based split
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()

        # Scale features (fit on train only)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_df[feature_cols])
        val_scaled = scaler.transform(val_df[feature_cols])

        train_targets = train_df['Target'].values
        val_targets = val_df['Target'].values

        # Create sequences
        X_train, y_train = create_sequences(train_scaled, train_targets, seq_length)
        X_val, y_val = create_sequences(val_scaled, val_targets, seq_length)

        if len(X_train) > 0:
            all_X_train.append(X_train)
            all_y_train.append(y_train)
        if len(X_val) > 0:
            all_X_val.append(X_val)
            all_y_val.append(y_val)

    # Combine all tickers
    X_train = np.concatenate(all_X_train, axis=0)
    y_train = np.concatenate(all_y_train, axis=0)
    X_val = np.concatenate(all_X_val, axis=0)
    y_val = np.concatenate(all_y_val, axis=0)

    console.print(f"[cyan]Train samples: {len(X_train)}, Val samples: {len(X_val)}[/cyan]")

    # Create datasets and loaders
    train_ds = PriceDataset(X_train, y_train)
    val_ds = PriceDataset(X_val, y_val)

    # Use more workers on cloud
    num_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    input_size = X_train.shape[2]

    return train_loader, val_loader, input_size


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main(data_dir="data/processed/price_prediction",
         output_dir="models",
         epochs=50,
         batch_size=256,
         lr=1e-3,
         hidden_size=128,
         num_layers=2):
    """Main training function"""

    console.print("[bold green]=" * 60)
    console.print("[bold green]InvestLLM Cloud GPU Training")
    console.print("[bold green]=" * 60)

    # Set random seed
    pl.seed_everything(42)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[bold cyan]GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)[/bold cyan]")
        accelerator = "cuda"
    else:
        console.print("[yellow]No GPU detected, using CPU[/yellow]")
        accelerator = "cpu"

    # Prepare data
    console.print("\n[bold]Preparing Data...[/bold]")
    train_loader, val_loader, input_size = prepare_data(
        data_dir, batch_size=batch_size
    )

    # Create model
    console.print(f"\n[bold]Creating Model (Input Size: {input_size})...[/bold]")
    model = PricePredictionLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        learning_rate=lr
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"Total Parameters: {total_params:,}")
    console.print(f"Trainable Parameters: {trainable_params:,}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename='investllm-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop],
        accelerator=accelerator,
        devices=1,
        precision="16-mixed" if accelerator == "cuda" else 32,  # Mixed precision for GPU
        log_every_n_steps=50,
        enable_progress_bar=True
    )

    # Train
    console.print("\n[bold green]Starting Training...[/bold green]")
    trainer.fit(model, train_loader, val_loader)

    # Results
    console.print("\n[bold green]=" * 60)
    console.print("[bold green]Training Complete!")
    console.print("[bold green]=" * 60)
    console.print(f"Best model: {checkpoint_callback.best_model_path}")
    console.print(f"Best val_loss: {checkpoint_callback.best_model_score:.6f}")

    # Save final model
    final_path = output_path / "investllm_final.ckpt"
    trainer.save_checkpoint(final_path)
    console.print(f"Final model saved: {final_path}")

    return checkpoint_callback.best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InvestLLM Cloud GPU Training")
    parser.add_argument("--data_dir", type=str, default="data/processed/price_prediction",
                        help="Directory containing processed parquet files")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size (increase for larger GPU memory)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of LSTM layers")

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
