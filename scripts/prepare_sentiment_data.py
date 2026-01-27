#!/usr/bin/env python3
"""
Prepare Sentiment Training Data from FinGPT Datasets
Combines multiple datasets and creates train/val/test splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

# Settings
DATA_DIR = Path("data/fingpt/datasets")
OUTPUT_DIR = Path("data/sentiment")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sentiment label mapping
LABEL_MAP = {
    "positive": 2,
    "neutral": 1, 
    "negative": 0,
    # Handle variations
    "bullish": 2,
    "bearish": 0,
    "mildly positive": 2,
    "mildly negative": 0,
    "moderately positive": 2,
    "moderately negative": 0,
    "strong positive": 2,
    "strong negative": 0,
}

def normalize_label(label: str) -> int:
    """Convert text label to numeric."""
    label = str(label).lower().strip()
    return LABEL_MAP.get(label, 1)  # Default to neutral

def load_sentiment_datasets() -> pd.DataFrame:
    """Load and combine all sentiment datasets."""
    all_data = []
    
    # 1. FinGPT Sentiment Training (main dataset)
    sent_train = DATA_DIR / "fingpt-sentiment-train.parquet"
    if sent_train.exists():
        df = pd.read_parquet(sent_train)
        print(f"Loaded fingpt-sentiment-train: {len(df)} samples")
        # Extract text and label
        if 'input' in df.columns and 'output' in df.columns:
            df_clean = pd.DataFrame({
                'text': df['input'].apply(lambda x: x.split('Answer:')[0].replace('Instruction: What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.', '').replace('Input:', '').strip() if isinstance(x, str) else ''),
                'label': df['output'].apply(normalize_label),
                'source': 'fingpt-sentiment-train'
            })
            all_data.append(df_clean)
    
    # 2. FinGPT Sentiment Classification
    sent_cls = DATA_DIR / "fingpt-sentiment-cls.parquet"
    if sent_cls.exists():
        df = pd.read_parquet(sent_cls)
        print(f"Loaded fingpt-sentiment-cls: {len(df)} samples")
        if 'input' in df.columns and 'output' in df.columns:
            df_clean = pd.DataFrame({
                'text': df['input'].apply(lambda x: x.split('Answer:')[0].replace('Instruction:', '').replace('Input:', '').strip()[:500] if isinstance(x, str) else ''),
                'label': df['output'].apply(normalize_label),
                'source': 'fingpt-sentiment-cls'
            })
            all_data.append(df_clean)
    
    # 3. FinGPT Headlines
    headlines = DATA_DIR / "fingpt-headline.parquet"
    if headlines.exists():
        df = pd.read_parquet(headlines)
        print(f"Loaded fingpt-headline: {len(df)} samples")
        if 'input' in df.columns and 'output' in df.columns:
            df_clean = pd.DataFrame({
                'text': df['input'].apply(lambda x: str(x)[:500] if x else ''),
                'label': df['output'].apply(normalize_label),
                'source': 'fingpt-headline'
            })
            all_data.append(df_clean)
    
    # Combine all datasets
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        # Remove empty texts
        combined = combined[combined['text'].str.len() > 10]
        # Remove duplicates
        combined = combined.drop_duplicates(subset=['text'])
        return combined
    return pd.DataFrame()

def create_splits(df: pd.DataFrame, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create train/val/test splits."""
    # First split: train + (val+test)
    train_df, temp_df = train_test_split(
        df, train_size=train_ratio, random_state=42, stratify=df['label']
    )
    
    # Second split: val + test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, train_size=val_ratio_adjusted, random_state=42, stratify=temp_df['label']
    )
    
    return train_df, val_df, test_df

def main():
    print("=" * 60)
    print("Preparing Sentiment Training Data")
    print("=" * 60)
    
    # Load datasets
    df = load_sentiment_datasets()
    print(f"\nTotal combined samples: {len(df):,}")
    
    # Show label distribution
    print("\nLabel Distribution:")
    for label, name in [(0, "Negative"), (1, "Neutral"), (2, "Positive")]:
        count = len(df[df['label'] == label])
        print(f"  {name}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Create splits
    train_df, val_df, test_df = create_splits(df)
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    
    # Save to parquet
    train_df.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val_df.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test_df.to_parquet(OUTPUT_DIR / "test.parquet", index=False)
    
    # Also save as CSV for easy viewing
    train_df.head(1000).to_csv(OUTPUT_DIR / "train_sample.csv", index=False)
    
    # Save metadata
    metadata = {
        "total_samples": len(df),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "label_map": {"negative": 0, "neutral": 1, "positive": 2},
        "sources": list(df['source'].unique())
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Data saved to: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
