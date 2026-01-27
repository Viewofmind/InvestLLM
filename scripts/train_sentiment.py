#!/usr/bin/env python3
"""
Train DistilBERT Sentiment Classifier for Financial News
Runs on CPU - optimized for MacBook without GPU.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

# Settings
DATA_DIR = Path("data/sentiment")
MODEL_DIR = Path("models/sentiment")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Force CPU
device = torch.device("cpu")
print(f"Using device: {device}")


class SentimentDataset(Dataset):
    """Dataset for sentiment classification."""
    
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
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_data(tokenizer, batch_size=16):
    """Load train/val/test datasets."""
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    
    train_dataset = SentimentDataset(
        train_df['text'].tolist(),
        train_df['label'].tolist(),
        tokenizer
    )
    val_dataset = SentimentDataset(
        val_df['text'].tolist(),
        val_df['label'].tolist(),
        tokenizer
    )
    test_dataset = SentimentDataset(
        test_df['text'].tolist(),
        test_df['label'].tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def train_epoch(model, dataloader, optimizer, scheduler):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    actuals = []
    
    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())
        actuals.extend(labels.cpu().numpy())
        
        progress.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(actuals, predictions)
    
    return avg_loss, accuracy


def evaluate(model, dataloader):
    """Evaluate on validation/test set."""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, average='weighted')
    
    return accuracy, f1, predictions, actuals


def main(args):
    print("=" * 60)
    print("Training DistilBERT Sentiment Classifier")
    print("=" * 60)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader = load_data(tokenizer, batch_size=args.batch_size)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Load model
    print("\nLoading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=3  # negative, neutral, positive
    )
    model.to(device)
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        val_acc, val_f1, _, _ = evaluate(model, val_loader)
        print(f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(MODEL_DIR / "best_model")
            tokenizer.save_pretrained(MODEL_DIR / "best_model")
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
    
    # Final evaluation on test set
    print("\n--- Final Evaluation on Test Set ---")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR / "best_model")
    model.to(device)
    
    test_acc, test_f1, preds, actuals = evaluate(model, test_loader)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    
    # Classification report
    label_names = ['Negative', 'Neutral', 'Positive']
    print("\nClassification Report:")
    print(classification_report(actuals, preds, target_names=label_names))
    
    # Save metrics
    metrics = {
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "best_val_accuracy": best_val_acc,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✅ Training complete! Model saved to: {MODEL_DIR / 'best_model'}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    
    main(args)
