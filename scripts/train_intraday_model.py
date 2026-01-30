"""
InvestLLM Intraday Model Training Script
========================================
Train LSTM + Attention model for intraday trading signals.

Usage (Local):
    python scripts/train_intraday_model.py \
        --data data/intraday_features/train_features.parquet \
        --epochs 100

Usage (Azure GPU):
    python scripts/train_intraday_model.py \
        --data data/intraday_features/train_features.parquet \
        --epochs 100 \
        --gpus 1 \
        --batch-size 512

Features:
- Automatic GPU detection
- Mixed precision training (FP16)
- Early stopping
- Model checkpointing
- WandB logging (optional)
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from investllm.models.intraday_model import (
    IntradayLSTM,
    IntradayDataset,
    create_data_loaders
)


def load_feature_columns(path: str) -> list:
    """Load feature column names from file"""
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def prepare_data(
    data_path: str,
    feature_columns: list = None,
    val_split: float = 0.2,
    batch_size: int = 256,
    sequence_length: int = 60,
    num_workers: int = 4
):
    """Prepare data loaders for training"""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    print(f"Loaded {len(df):,} rows, {df['SYMBOL'].nunique()} symbols")

    # Auto-detect feature columns if not provided
    if feature_columns is None:
        exclude_cols = [
            'SYMBOL', 'TOKEN', 'EXCHANGE', 'MARKET CAP', 'TIME',
            'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE', 'VOLUME',
            'target_direction', 'target_up', 'target_return', 'date', 'vwap',
            'obv', 'ad_line', 'hour', 'minute', 'day_of_week',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'ema_5', 'ema_10', 'ema_20', 'ema_50',
            'bb_upper_10', 'bb_lower_10', 'bb_upper_20', 'bb_lower_20',
            'atr_7', 'atr_14', 'volume_sma_5', 'volume_sma_10', 'volume_sma_20',
            'price_min_5', 'price_max_5', 'price_min_10', 'price_max_10',
            'price_min_20', 'price_max_20', 'price_min_40', 'price_max_40',
            'price_min_60', 'price_max_60', 'returns_skew_5', 'returns_skew_10',
            'returns_skew_20', 'returns_skew_40', 'returns_skew_60',
            'returns_kurt_5', 'returns_kurt_10', 'returns_kurt_20',
            'returns_kurt_40', 'returns_kurt_60'
        ]
        feature_columns = [c for c in df.columns if c not in exclude_cols]

    print(f"Using {len(feature_columns)} features")

    # Handle NaN/Inf values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_columns + ['target_direction', 'target_return'])
    print(f"After cleaning: {len(df):,} rows")

    # Create data loaders
    train_loader, val_loader, scaler, class_weights = create_data_loaders(
        df,
        feature_columns,
        batch_size=batch_size,
        sequence_length=sequence_length,
        val_split=val_split,
        num_workers=num_workers
    )

    print(f"\nTrain samples: {len(train_loader.dataset):,}")
    print(f"Val samples: {len(val_loader.dataset):,}")
    print(f"Class weights: {class_weights.numpy()}")

    return train_loader, val_loader, scaler, class_weights, len(feature_columns)


def train(args):
    """Main training function"""
    print("="*60)
    print("InvestLLM Intraday Model Training")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load feature columns
    feature_columns = None
    if args.features:
        feature_columns = load_feature_columns(args.features)

    # Prepare data
    train_loader, val_loader, scaler, class_weights, input_dim = prepare_data(
        args.data,
        feature_columns=feature_columns,
        val_split=args.val_split,
        batch_size=args.batch_size,
        sequence_length=args.seq_length,
        num_workers=args.num_workers
    )

    # Move class weights to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_weights = class_weights.to(device)

    # Create model
    model = IntradayLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        class_weights=class_weights
    )

    print(f"\nModel architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  LSTM layers: {args.num_layers}")
    print(f"  Attention heads: {args.num_heads}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=args.patience,
            mode='min',
            verbose=True
        ),
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename='intraday-{epoch:02d}-{val_loss:.4f}-{val_accuracy:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        RichProgressBar()
    ]

    # Logger
    if args.wandb:
        logger = WandbLogger(
            project='investllm-intraday',
            name=f'lstm-{args.hidden_dim}-{args.num_layers}L',
            log_model=True
        )
    else:
        logger = TensorBoardLogger(
            save_dir=args.log_dir,
            name='intraday_model'
        )

    # Trainer
    trainer_kwargs = {
        'max_epochs': args.epochs,
        'callbacks': callbacks,
        'logger': logger,
        'log_every_n_steps': 50,
        'gradient_clip_val': 1.0,
        'deterministic': True,
        'enable_progress_bar': True,
    }

    # GPU settings
    if torch.cuda.is_available() and args.gpus > 0:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = args.gpus
        if args.fp16:
            trainer_kwargs['precision'] = '16-mixed'
    else:
        trainer_kwargs['accelerator'] = 'cpu'

    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)

    trainer.fit(model, train_loader, val_loader)

    # Save final model
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"\nBest model saved to: {best_model_path}")

    # Save scaler
    scaler_path = Path(args.checkpoint_dir) / 'scaler.pkl'
    import pickle
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Evaluate best model
    print("\n" + "="*60)
    print("Final Evaluation")
    print("="*60)

    best_model = IntradayLSTM.load_from_checkpoint(best_model_path)
    best_model.eval()

    # Run validation
    results = trainer.validate(best_model, val_loader)
    print(f"\nValidation Results:")
    for k, v in results[0].items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return best_model, scaler


def main():
    parser = argparse.ArgumentParser(description='Train intraday trading model')

    # Data
    parser.add_argument('--data', type=str,
                       default='data/intraday_features/train_features.parquet',
                       help='Path to training data')
    parser.add_argument('--features', type=str, default=None,
                       help='Path to feature columns file')
    parser.add_argument('--val-split', type=float, default=0.2,
                       help='Validation split ratio')

    # Model architecture
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='LSTM hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--num-heads', type=int, default=4,
                       help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--seq-length', type=int, default=60,
                       help='Sequence length (lookback)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')

    # Hardware
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of GPUs (0 for CPU)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')

    # Logging
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/intraday',
                       help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='lightning_logs',
                       help='TensorBoard log directory')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases logging')

    args = parser.parse_args()

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Train
    train(args)


if __name__ == '__main__':
    main()
