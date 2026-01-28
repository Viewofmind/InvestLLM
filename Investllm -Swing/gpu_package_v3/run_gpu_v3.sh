#!/bin/bash
# ============================================================
# InvestLLM V3 - GPU Training & Backtest
# Run on RunPod or any GPU machine
# ============================================================

echo "============================================================"
echo "InvestLLM Swing Trading V3 - GPU Pipeline"
echo "============================================================"
echo ""

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
    DEVICE="cuda"
else
    echo "No GPU detected, using CPU"
    DEVICE="cpu"
fi

echo ""
echo "============================================================"
echo "Step 1: Install Dependencies"
echo "============================================================"

pip install torch pandas numpy scikit-learn pyarrow --quiet

echo ""
echo "============================================================"
echo "Step 2: Train Ensemble Models (3 models)"
echo "============================================================"
echo ""

python train_ensemble.py \
    --features swing_features/all_swing_features.parquet \
    --output ensemble_models/ \
    --device $DEVICE \
    --n-models 3 \
    --epochs 25 \
    --seeds 42 142 242

echo ""
echo "============================================================"
echo "Step 3: Run V3 Backtest with Ensemble"
echo "============================================================"
echo ""

# Get all model files
MODEL_FILES=$(ls ensemble_models/model_seed_*.pt 2>/dev/null | tr '\n' ' ')

if [ -z "$MODEL_FILES" ]; then
    echo "No ensemble models found, using single model if available"
    MODEL_FILES="swing_models/trained/best_model.pt"
fi

echo "Models: $MODEL_FILES"

python swing_backtester_v3.py \
    --model $MODEL_FILES \
    --scaler ensemble_models/scaler.pkl \
    --features swing_features/ \
    --capital 100000 \
    --output reports/swing_backtest_v3/ \
    --device $DEVICE \
    --temperature 0.5 \
    --max-positions 10

echo ""
echo "============================================================"
echo "V3 Pipeline Complete!"
echo "============================================================"
echo ""
echo "Files created:"
echo "  ensemble_models/"
echo "    - model_seed_42.pt"
echo "    - model_seed_142.pt"
echo "    - model_seed_242.pt"
echo "    - scaler.pkl"
echo "    - ensemble_config.json"
echo ""
echo "  reports/swing_backtest_v3/"
echo "    - all_trades_v3.csv"
echo "    - equity_curve_v3.csv"
echo "    - exit_reason_analysis_v3.csv"
echo ""
echo "Download these files to your local machine!"
