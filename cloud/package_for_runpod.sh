#!/bin/bash
# Package InvestLLM for RunPod GPU training

echo "Creating investllm_ensemble.zip for RunPod..."

# Create zip with essential files
zip -r investllm_ensemble.zip \
    cloud/train_ensemble_runpod.py \
    cloud/requirements_sentiment.txt \
    data/processed/price_prediction/*.parquet \
    models/sentiment/sentiment_model_final/ \
    investllm/models/ \
    investllm/strategies/ \
    models/price_prediction/*.py \
    -x "*.pyc" -x "__pycache__/*" -x "*.git/*"

echo ""
echo "Package created: investllm_ensemble.zip"
ls -lh investllm_ensemble.zip
echo ""
echo "Upload to RunPod and run:"
echo "  unzip investllm_ensemble.zip"
echo "  pip install torch pytorch-lightning transformers pandas numpy scikit-learn pyarrow"
echo "  python cloud/train_ensemble_runpod.py"
