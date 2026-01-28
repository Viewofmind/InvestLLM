"""
Ensemble Model Training for V3 Strategy
Train multiple models with different seeds for ensemble prediction

Usage on GPU:
    python train_ensemble.py --device cuda --n-models 3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import argparse
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SwingTradingDataset(Dataset):
    """Dataset for swing trading with proper time series handling"""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 timestamps: np.ndarray, symbols: np.ndarray = None,
                 sequence_length=30):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.timestamps = timestamps
        self.symbols = symbols
        self.sequence_length = sequence_length
        self.valid_indices = self._build_valid_indices()

    def _build_valid_indices(self):
        valid = []
        if self.symbols is None:
            for i in range(len(self.features) - self.sequence_length):
                valid.append(i)
        else:
            for i in range(len(self.features) - self.sequence_length):
                seq_symbols = self.symbols[i:i + self.sequence_length + 1]
                if len(set(seq_symbols)) == 1:
                    valid.append(i)
        return valid

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        X = self.features[real_idx:real_idx + self.sequence_length]
        y = self.targets[real_idx + self.sequence_length]
        return X, y


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


def load_and_prepare_data(feature_file: str, target_horizon: int = 5):
    """Load and prepare data with chronological splits"""
    print(f"Loading data from {feature_file}...")
    df = pd.read_parquet(feature_file)

    # Sort by symbol and timestamp
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)

    # Create target
    df['target'] = df.groupby('symbol')['close'].shift(-target_horizon)
    df['target'] = (df['target'] - df['close']) / df['close']
    df = df.dropna(subset=['target'])

    # Feature columns
    exclude_cols = ['symbol', 'timestamp', 'target', 'open', 'high', 'low', 'close', 'volume', 'exchange']
    feature_cols = [col for col in df.columns if col not in exclude_cols
                   and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

    print(f"Features: {len(feature_cols)}")
    print(f"Total samples: {len(df):,}")

    # Chronological split
    unique_dates = sorted(df['timestamp'].unique())
    n_dates = len(unique_dates)
    train_end = unique_dates[int(n_dates * 0.7)]
    val_end = unique_dates[int(n_dates * 0.85)]

    train_df = df[df['timestamp'] < train_end]
    val_df = df[(df['timestamp'] >= train_end) & (df['timestamp'] < val_end)]
    test_df = df[df['timestamp'] >= val_end]

    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Scale features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_df[feature_cols])
    val_features = scaler.transform(val_df[feature_cols])
    test_features = scaler.transform(test_df[feature_cols])

    # Create datasets
    train_dataset = SwingTradingDataset(
        train_features, train_df['target'].values,
        train_df['timestamp'].values, train_df['symbol'].values
    )
    val_dataset = SwingTradingDataset(
        val_features, val_df['target'].values,
        val_df['timestamp'].values, val_df['symbol'].values
    )
    test_dataset = SwingTradingDataset(
        test_features, test_df['target'].values,
        test_df['timestamp'].values, test_df['symbol'].values
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader, len(feature_cols), scaler


def train_single_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    device: str,
    seed: int,
    epochs: int = 25,
    patience: int = 5
) -> nn.Module:
    """Train a single model with given seed"""
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'='*50}")
    print(f"Training Model with Seed {seed}")
    print(f"{'='*50}")

    model = SwingLSTM(input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct_direction = 0
        total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                val_loss += criterion(pred, y).item()

                # Direction accuracy
                pred_dir = (pred > 0).float()
                actual_dir = (y > 0).float()
                correct_direction += (pred_dir == actual_dir).sum().item()
                total += len(y)

        val_loss /= len(val_loader)
        direction_acc = correct_direction / total

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | Dir Acc: {direction_acc*100:.1f}%")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    model.load_state_dict(best_state)
    return model


def evaluate_ensemble(models: list, test_loader: DataLoader, device: str):
    """Evaluate ensemble on test set"""
    print("\n" + "="*50)
    print("Evaluating Ensemble")
    print("="*50)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)

            # Get predictions from all models
            model_preds = []
            for model in models:
                model.eval()
                pred = model(X)
                model_preds.append(pred.cpu().numpy())

            # Ensemble average
            ensemble_pred = np.mean(model_preds, axis=0)
            all_predictions.extend(ensemble_pred)
            all_targets.extend(y.numpy())

    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Direction accuracy
    pred_dir = predictions > 0
    actual_dir = targets > 0
    direction_acc = (pred_dir == actual_dir).mean()

    # MSE
    mse = ((predictions - targets) ** 2).mean()

    # Model agreement (model_preds already numpy arrays)
    std_pred = np.std(model_preds, axis=0)
    avg_agreement = 1 - np.mean(std_pred) * 5

    print(f"Direction Accuracy: {direction_acc*100:.2f}%")
    print(f"MSE: {mse:.6f}")
    print(f"Avg Model Agreement: {avg_agreement*100:.1f}%")

    return direction_acc, mse


def main():
    parser = argparse.ArgumentParser(description='Train Ensemble Models')
    parser.add_argument('--features', type=str, default='swing_features/all_swing_features.parquet')
    parser.add_argument('--output', type=str, default='ensemble_models/')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n-models', type=int, default=3, help='Number of models in ensemble')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Random seeds for each model')

    args = parser.parse_args()

    print("="*60)
    print("ENSEMBLE TRAINING V3")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Number of models: {args.n_models}")

    # Load data
    train_loader, val_loader, test_loader, num_features, scaler = load_and_prepare_data(
        args.features
    )

    # Generate seeds if not provided
    if args.seeds is None:
        args.seeds = [42 + i * 100 for i in range(args.n_models)]

    # Train ensemble
    models = []
    for i, seed in enumerate(args.seeds[:args.n_models]):
        model = train_single_model(
            train_loader, val_loader, num_features,
            args.device, seed, args.epochs
        )
        models.append(model)

    # Evaluate ensemble
    direction_acc, mse = evaluate_ensemble(models, test_loader, args.device)

    # Save models
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    for i, model in enumerate(models):
        model_path = output_path / f'model_seed_{args.seeds[i]}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'seed': args.seeds[i],
            'direction_accuracy': direction_acc
        }, model_path)
        print(f"Saved: {model_path}")

    # Save scaler
    scaler_path = output_path / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {scaler_path}")

    # Save ensemble config
    config = {
        'n_models': len(models),
        'seeds': args.seeds[:args.n_models],
        'direction_accuracy': float(direction_acc),
        'mse': float(mse),
        'input_dim': num_features
    }

    import json
    with open(output_path / 'ensemble_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved to: {output_path}")
    print(f"Ensemble Direction Accuracy: {direction_acc*100:.2f}%")


if __name__ == '__main__':
    main()
