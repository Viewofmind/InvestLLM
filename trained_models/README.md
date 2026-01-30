# Trained Models Directory

This directory contains trained model checkpoints and artifacts.

## 📁 Directory Structure

```
trained_models/
├── intraday_4years/           # Intraday trading model (5-min bars)
│   ├── *.ckpt                 # PyTorch Lightning checkpoint (16 MB)
│   ├── scaler.pkl             # Feature scaler (2.5 KB)
│   ├── feature_names.txt      # List of 89 features
│   └── training_4years.log    # Training metrics
│
└── sentiment/                 # Sentiment analysis models
    └── finbert/               # FinBERT fine-tuned model
```

## 🚫 Git Exclusion

Model files are excluded from Git due to size:
- `*.ckpt` files (checkpoint files)
- `*.pkl` files (pickle files)

## 📥 How to Get Model Files

### Option 1: Download from Release

Download pre-trained models from GitHub Releases:
```bash
# Download intraday model
wget https://github.com/Viewofmind/InvestLLM/releases/download/v1.0/intraday_model.tar.gz
tar -xzf intraday_model.tar.gz -C trained_models/
```

### Option 2: Train Your Own

**Intraday Model:**
```bash
# See: docs/INTRADAY_MODEL_README.md

# 1. Collect data
python scripts/kite_data_collector.py --symbols NIFTY100 --start-date 2021-01-01

# 2. Generate features
python scripts/intraday_feature_engineering.py

# 3. Train on GPU (RunPod recommended)
python scripts/train_intraday_model.py
```

**Cost**: ~$1-2 on RunPod RTX 4090 (1-2 hours)

### Option 3: Request Access

Contact the repository owner for access to pre-trained models.

## 📊 Model Specifications

### Intraday Model v1.0 (Jan 2026)

| Property | Value |
|----------|-------|
| **File** | `intraday_4years/intraday-epoch=03-val_loss=1.0177-val_accuracy=0.5612.ckpt` |
| **Size** | 16 MB |
| **Architecture** | LSTM + Multi-Head Attention |
| **Parameters** | 1,407,493 |
| **Training Data** | 7M samples, 97 stocks, 4 years |
| **Validation Accuracy** | 56.12% |
| **Input Features** | 89 technical indicators |
| **Output Classes** | 3 (BUY/SELL/HOLD) |

**Documentation**: [docs/INTRADAY_MODEL_README.md](../docs/INTRADAY_MODEL_README.md)

## 🔐 Model Checksums

Verify downloaded models:

```bash
# Intraday model checkpoint
sha256sum trained_models/intraday_4years/*.ckpt
# Expected: (will be updated after release)

# Scaler
sha256sum trained_models/intraday_4years/scaler.pkl
# Expected: (will be updated after release)
```

## ⚠️ Important Notes

1. **Model files are large** - Don't commit to Git
2. **Use Git LFS** if you must track model versions
3. **Store models externally** (Google Drive, S3, etc.) for team sharing
4. **Update checksums** when releasing new model versions

## 📞 Support

For issues with model files:
- Check [docs/INTRADAY_MODEL_README.md](../docs/INTRADAY_MODEL_README.md)
- Review training logs in `training_4years.log`
- Open an issue on GitHub
