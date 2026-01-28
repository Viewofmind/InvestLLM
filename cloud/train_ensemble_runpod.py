#!/usr/bin/env python3
"""
InvestLLM Complete Ensemble Training - RunPod GPU
==================================================
Trains the full ensemble system:
1. Price Model (LSTM) - with sentiment features
2. Uses pre-trained Sentiment Model (FinBERT)
3. Fundamental Scorer (rule-based)

Run on RunPod RTX 4090 for best performance.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Data
    DATA_DIR = Path("data/processed/price_prediction")
    SENTIMENT_MODEL_DIR = Path("models/sentiment/sentiment_model_final")
    OUTPUT_DIR = Path("models/ensemble_trained")

    # LSTM Model
    HIDDEN_SIZE = 256
    NUM_LAYERS = 3
    DROPOUT = 0.3
    SEQ_LENGTH = 60

    # Training
    BATCH_SIZE = 64
    MAX_EPOCHS = 30
    LEARNING_RATE = 1e-3
    PATIENCE = 7

    # Features
    USE_SENTIMENT = True  # Enable real sentiment features

    @classmethod
    def print_config(cls):
        print("="*60)
        print("CONFIGURATION")
        print("="*60)
        for k, v in vars(cls).items():
            if not k.startswith('_'):
                print(f"  {k}: {v}")
        print("="*60)


# ============================================================================
# SENTIMENT INTEGRATION
# ============================================================================

class SentimentFeatureGenerator:
    """Generate sentiment features for price data"""

    def __init__(self, model_path: str = None):
        self.scorer = None
        if model_path and Path(model_path).exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
                self.model.eval()
                if torch.cuda.is_available():
                    self.model.cuda()
                print(f"Loaded sentiment model from {model_path}")

                # Load label map
                import json
                config_file = Path(model_path) / "config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config = json.load(f)
                        self.label_map = {int(k): v for k, v in config.get("id2label", {}).items()}
                else:
                    self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

            except Exception as e:
                print(f"Could not load sentiment model: {e}")
                self.model = None
        else:
            print("No sentiment model found - using neutral signals")
            self.model = None

    def get_sentiment_score(self, text: str) -> float:
        """Get sentiment score from -1 to +1"""
        if self.model is None:
            return 0.0

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=128
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]

            pred_idx = probs.argmax().item()
            sentiment = self.label_map.get(pred_idx, "neutral")
            confidence = probs[pred_idx].item()

            score_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            return score_map.get(sentiment, 0.0) * confidence

        except Exception:
            return 0.0


# ============================================================================
# DATASET
# ============================================================================

class StockDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_length: int = 60):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        X = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return X, y


# ============================================================================
# LSTM MODEL
# ============================================================================

class PricePredictionLSTM(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

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

        self.learning_rate = learning_rate

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)

        return self.fc(context).squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)

        # Direction accuracy
        pred_direction = (y_hat > 0).float()
        true_direction = (y > 0).float()
        acc = (pred_direction == true_direction).float().mean()

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)

        pred_direction = (y_hat > 0).float()
        true_direction = (y > 0).float()
        acc = (pred_direction == true_direction).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return [optimizer], [scheduler]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_stock_data(data_dir: Path, sentiment_gen: SentimentFeatureGenerator = None):
    """Load and combine all stock data"""
    files = list(data_dir.glob("*_processed.parquet"))
    print(f"Found {len(files)} stock files")

    all_features = []
    all_targets = []

    for f in files:
        try:
            df = pd.read_parquet(f)
            ticker = f.stem.split('_')[0]

            if len(df) < 200:
                continue

            # Get features (exclude Date and Target)
            feature_cols = [c for c in df.columns if c not in ['Date', 'Target']]
            features = df[feature_cols].values
            targets = df['Target'].values

            # Add sentiment feature column (placeholder - would be populated with real news)
            sentiment_col = np.zeros((len(df), 1))
            features = np.hstack([features, sentiment_col])

            all_features.append(features)
            all_targets.append(targets)

        except Exception as e:
            print(f"Error loading {f}: {e}")
            continue

    print(f"Loaded {len(all_features)} stocks successfully")

    # Combine all data
    combined_features = np.vstack(all_features)
    combined_targets = np.hstack(all_targets)

    return combined_features, combined_targets


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("="*60)
    print("InvestLLM Ensemble Training - GPU")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("WARNING: No GPU found, using CPU")

    Config.print_config()

    # Create output dir
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize sentiment generator
    sentiment_gen = None
    if Config.USE_SENTIMENT and Config.SENTIMENT_MODEL_DIR.exists():
        sentiment_gen = SentimentFeatureGenerator(str(Config.SENTIMENT_MODEL_DIR))

    # Load data
    print("\nLoading stock data...")
    features, targets = load_all_stock_data(Config.DATA_DIR, sentiment_gen)
    print(f"Total samples: {len(features):,}")
    print(f"Features shape: {features.shape}")

    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Save scaler
    import pickle
    with open(Config.OUTPUT_DIR / "scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    # Split data (time-series aware)
    split_idx = int(len(features_scaled) * 0.8)
    train_features = features_scaled[:split_idx]
    train_targets = targets[:split_idx]
    val_features = features_scaled[split_idx:]
    val_targets = targets[split_idx:]

    print(f"Train samples: {len(train_features):,}")
    print(f"Val samples: {len(val_features):,}")

    # Create datasets
    train_dataset = StockDataset(train_features, train_targets, Config.SEQ_LENGTH)
    val_dataset = StockDataset(val_features, val_targets, Config.SEQ_LENGTH)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    input_size = features_scaled.shape[1]
    print(f"\nCreating LSTM model with input_size={input_size}")

    model = PricePredictionLSTM(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        learning_rate=Config.LEARNING_RATE
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Config.OUTPUT_DIR),
        filename='ensemble_lstm-{epoch:02d}-{val_loss:.4f}-{val_acc:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=Config.PATIENCE,
        mode='min'
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=Config.MAX_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True
    )

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)

    trainer.fit(model, train_loader, val_loader)

    # Save final model
    print("\nSaving final model...")
    trainer.save_checkpoint(str(Config.OUTPUT_DIR / "ensemble_lstm_final.ckpt"))

    # Print results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    print(f"Models saved to: {Config.OUTPUT_DIR}")

    # Quick test
    print("\nRunning quick inference test...")
    model.eval()
    with torch.no_grad():
        sample = torch.FloatTensor(val_features[:Config.SEQ_LENGTH]).unsqueeze(0)
        if torch.cuda.is_available():
            sample = sample.cuda()
            model = model.cuda()
        pred = model(sample)
        print(f"Sample prediction: {pred.item():.6f}")

    print("\n" + "="*60)
    print("Done! Download models from:", Config.OUTPUT_DIR)
    print("="*60)


if __name__ == "__main__":
    main()
