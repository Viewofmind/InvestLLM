# InvestLLM Cloud Training on RunPod

## Quick Start Guide

### Step 1: Create RunPod Account
1. Go to [runpod.io](https://runpod.io)
2. Sign up and add credits ($10-20 is enough for training)

### Step 2: Launch a GPU Pod

1. Click **"Deploy"** or **"GPU Pods"**
2. Select a GPU:
   - **Budget:** RTX 3090 (~$0.30/hr) - Good for this model
   - **Fast:** RTX 4090 (~$0.50/hr) - 2x faster
   - **Pro:** A100 (~$1.50/hr) - Best for large models

3. Select Template:
   - Choose **"RunPod Pytorch 2.0"** or **"PyTorch"**
   - This has CUDA pre-installed

4. Set Volume:
   - Add 20GB volume for data storage

5. Click **"Deploy"**

### Step 3: Upload Training Data

**Option A: Upload via Web UI**
1. Click on your pod → "Connect" → "File Browser"
2. Upload `investllm_cloud.zip`
3. Extract in terminal

**Option B: Upload via SCP/SFTP**
```bash
# Get your pod's SSH command from RunPod UI
scp investllm_cloud.zip root@<pod-ip>:/workspace/
```

**Option C: Use runpodctl**
```bash
pip install runpodctl
runpodctl send investllm_cloud.zip
```

### Step 4: Run Training

1. Connect to pod terminal (Web Terminal or SSH)

2. Extract and setup:
```bash
cd /workspace
unzip investllm_cloud.zip
pip install -r cloud/requirements_cloud.txt
```

3. Start training:
```bash
python cloud/train_runpod.py \
    --epochs 50 \
    --batch_size 512 \
    --hidden_size 256 \
    --num_layers 3
```

### Step 5: Monitor Training

Training will show:
- GPU detected (e.g., "RTX 4090 (24.0 GB)")
- Training progress with val_loss and val_acc
- Checkpoints saved to `/workspace/models/`

### Step 6: Download Trained Model

After training completes:
```bash
# Compress the model
cd /workspace
zip -r trained_model.zip models/

# Download via runpodctl
runpodctl receive trained_model.zip
```

Or use the File Browser in RunPod UI.

---

## Recommended Training Configurations

### Quick Test (5-10 min)
```bash
python cloud/train_runpod.py --epochs 10 --batch_size 256
```

### Standard Training (30-60 min)
```bash
python cloud/train_runpod.py --epochs 50 --batch_size 512 --hidden_size 256
```

### Deep Training (2-3 hours)
```bash
python cloud/train_runpod.py \
    --epochs 100 \
    --batch_size 512 \
    --hidden_size 256 \
    --num_layers 3
```

---

## Cost Estimates

| GPU | Price/hr | 50 epochs time | Cost |
|-----|----------|----------------|------|
| RTX 3090 | $0.30 | ~45 min | ~$0.25 |
| RTX 4090 | $0.50 | ~25 min | ~$0.25 |
| A100 40GB | $1.50 | ~15 min | ~$0.40 |

**Total cost for full training: $0.25 - $0.50**

---

## Troubleshooting

### "CUDA out of memory"
Reduce batch size:
```bash
python cloud/train_runpod.py --batch_size 128
```

### "No parquet files found"
Check the data directory:
```bash
ls data/processed/price_prediction/
```

### Pod disconnected
Your training continues! Reconnect and check:
```bash
ls /workspace/models/
```

---

## After Training

1. Download the best checkpoint (lowest val_loss)
2. Copy to your local `models/price_prediction/` folder
3. Run backtest:
```bash
python scripts/strategy_backtester_smart.py
```

---

## Files Included

```
investllm_cloud.zip
├── cloud/
│   ├── train_runpod.py          # Training script
│   └── requirements_cloud.txt    # Dependencies
└── data/processed/price_prediction/
    ├── RELIANCE_processed.parquet
    ├── TCS_processed.parquet
    └── ... (98 stocks total)
```

---

**Happy Training!**
