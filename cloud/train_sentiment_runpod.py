#!/usr/bin/env python3
"""
InvestLLM Sentiment Model Training for RunPod
==============================================
Fine-tunes FinBERT/DistilBERT on financial sentiment data.

Usage:
    python train_sentiment_runpod.py --model finbert --epochs 5 --batch_size 32

Requirements:
    pip install torch transformers datasets pandas numpy scikit-learn accelerate rich
"""

import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from rich.console import Console
from rich.table import Table

console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_OPTIONS = {
    "finbert": "ProsusAI/finbert",
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "roberta": "roberta-base"
}

LABEL_MAP = {
    "positive": 2,
    "neutral": 1,
    "negative": 0
}

REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


# ============================================================================
# DATASET CLASS
# ============================================================================

class SentimentDataset(Dataset):
    """Dataset for sentiment classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_fingpt_data(data_dir):
    """Load FinGPT sentiment dataset"""
    console.print("[cyan]Loading FinGPT sentiment data...[/cyan]")

    # Look for parquet files
    data_path = Path(data_dir)

    # Try different possible locations
    possible_files = [
        data_path / "fingpt_sentiment.parquet",
        data_path / "fingpt-sentiment-train.parquet",
        data_path / "data/fingpt/datasets/fingpt-sentiment-train.parquet"
    ]

    df = None
    for file_path in possible_files:
        if file_path.exists():
            df = pd.read_parquet(file_path)
            console.print(f"[green]Loaded: {file_path}[/green]")
            break

    if df is None:
        # Try to load from HuggingFace
        console.print("[yellow]Local file not found, downloading from HuggingFace...[/yellow]")
        from datasets import load_dataset
        dataset = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
        df = dataset.to_pandas()

    console.print(f"[green]Loaded {len(df)} samples[/green]")
    return df


def prepare_data(df, tokenizer, test_size=0.15, val_size=0.15):
    """Prepare train/val/test splits"""

    # Get text and labels
    if 'input' in df.columns:
        texts = df['input'].tolist()
    elif 'text' in df.columns:
        texts = df['text'].tolist()
    else:
        raise ValueError("No text column found")

    if 'output' in df.columns:
        # Convert string labels to integers
        labels = df['output'].map(lambda x: LABEL_MAP.get(x.lower().strip(), 1)).tolist()
    elif 'label' in df.columns:
        labels = df['label'].tolist()
    else:
        raise ValueError("No label column found")

    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=42, stratify=labels
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
    )

    console.print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create datasets
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    val_dataset = SentimentDataset(X_val, y_val, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)

    return train_dataset, val_dataset, test_dataset


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'f1': f1
    }


# ============================================================================
# MAIN TRAINING
# ============================================================================

def main(
    model_name="finbert",
    data_dir="data",
    output_dir="models/sentiment",
    epochs=5,
    batch_size=32,
    learning_rate=2e-5,
    max_length=128
):
    """Main training function"""

    console.print("[bold green]=" * 60)
    console.print("[bold green]InvestLLM Sentiment Model Training")
    console.print("[bold green]=" * 60)

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        console.print(f"[bold cyan]GPU: {gpu_name}[/bold cyan]")
    else:
        console.print("[yellow]No GPU detected, using CPU[/yellow]")

    # Get model path
    model_path = MODEL_OPTIONS.get(model_name, model_name)
    console.print(f"[blue]Model: {model_path}[/blue]")

    # Load tokenizer and model
    console.print("\n[bold]Loading tokenizer and model...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,  # positive, neutral, negative
        ignore_mismatched_sizes=True
    )

    # Load data
    console.print("\n[bold]Loading data...[/bold]")
    df = load_fingpt_data(data_dir)

    # Prepare datasets
    console.print("\n[bold]Preparing datasets...[/bold]")
    train_dataset, val_dataset, test_dataset = prepare_data(df, tokenizer)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_path / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        warmup_ratio=0.1,
        logging_steps=100,
        fp16=device == "cuda",  # Mixed precision on GPU
        dataloader_num_workers=4,
        report_to="none"  # Disable wandb/tensorboard
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    console.print("\n[bold green]Starting Training...[/bold green]")
    trainer.train()

    # Evaluate on test set
    console.print("\n[bold]Evaluating on test set...[/bold]")
    test_results = trainer.evaluate(test_dataset)

    # Print results
    table = Table(title="Test Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in test_results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")

    console.print(table)

    # Save model
    final_path = output_path / "sentiment_model_final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    console.print(f"\n[bold green]Model saved to: {final_path}[/bold green]")

    # Generate predictions on test set for analysis
    console.print("\n[bold]Generating classification report...[/bold]")
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['Negative', 'Neutral', 'Positive']))

    return final_path


# ============================================================================
# INFERENCE HELPER
# ============================================================================

class SentimentScorer:
    """Helper class for sentiment scoring"""

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def score(self, text):
        """Score a single text, returns -1 to +1"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors='pt'
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Convert to -1 to +1 scale
        # negative=0, neutral=1, positive=2
        neg_prob = probs[0][0].item()
        neu_prob = probs[0][1].item()
        pos_prob = probs[0][2].item()

        # Sentiment score: weighted average
        score = (pos_prob * 1) + (neu_prob * 0) + (neg_prob * -1)

        return {
            'score': score,
            'positive': pos_prob,
            'neutral': neu_prob,
            'negative': neg_prob,
            'label': REVERSE_LABEL_MAP[probs.argmax().item()]
        }

    def score_batch(self, texts):
        """Score multiple texts"""
        return [self.score(text) for text in texts]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Sentiment Model")
    parser.add_argument("--model", type=str, default="finbert",
                        choices=list(MODEL_OPTIONS.keys()),
                        help="Model to fine-tune")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default="models/sentiment",
                        help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")

    args = parser.parse_args()

    main(
        model_name=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
