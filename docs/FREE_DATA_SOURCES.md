# FREE Data Sources Guide

## Overview

These FREE sources can save you significant time and money:

| Source | Data | Size | Use Case |
|--------|------|------|----------|
| **HuggingFace** | Pre-labeled sentiment news | 70K+ articles | **Sentiment training** |
| **Kaggle** | NIFTY 50 historical (2000-2021) | 50+ stocks, 20 years | **Price history** |
| **yfinance** | Current prices + fundamentals | 100+ stocks | **Recent data** |

---

## 1. HuggingFace Sentiment Data (GOLD! 🏆)

### Why This is Amazing

The `kdave/Indian_Financial_News` dataset has:
- **10,000+ Indian financial news articles**
- **Already labeled** as positive/negative/neutral
- **Saves weeks of manual labeling work**

### Download Commands

```bash
# Download all HuggingFace datasets
python scripts/download_hf_news.py --all

# Or just the Indian Financial News
python scripts/download_hf_news.py
```

### Available Datasets

| Dataset | Articles | Labels | Use |
|---------|----------|--------|-----|
| `kdave/Indian_Financial_News` | 10K+ | ✅ Yes | **Primary** |
| `FinGPT/fingpt-sentiment-train` | 50K+ | ✅ Yes | Augmentation |
| `twitter-financial-news-sentiment` | 10K+ | ✅ Yes | Social |
| `financial_phrasebank` | 5K+ | ✅ Yes | Academic |

### Output

```
data/
├── huggingface/
│   ├── indian_financial_news.parquet
│   ├── fingpt_sentiment.parquet
│   └── ...
└── processed/
    └── sentiment/
        └── sentiment_training_data.parquet  # Combined!
```

### What You Get

```python
# Sample data structure
{
    "text": "Reliance Industries Q3 profit rises 15% to Rs 18,549 crore",
    "label": "positive",
    "source": "hf_indian_financial_news"
}
```

---

## 2. Kaggle Historical Data

### Setup (One-time)

1. **Create Kaggle account**: https://www.kaggle.com
2. **Get API credentials**:
   - Go to kaggle.com/account
   - Click "Create New API Token"
   - Download `kaggle.json`
3. **Install credentials**:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Download Commands

```bash
# List available datasets
python scripts/download_kaggle_data.py --list

# Download NIFTY 50 data (2000-2021)
python scripts/download_kaggle_data.py

# Download all Indian stock datasets
python scripts/download_kaggle_data.py --all
```

### Available Datasets

| Dataset | Stocks | Years | Size |
|---------|--------|-------|------|
| `rohanrao/nifty50-stock-market-data` | 50 | 2000-2021 | 50 MB |
| `rohanrao/nifty-indices-dataset` | 10 indices | 2000-2021 | 10 MB |
| `sudalairajkumar/indian-stock-market-data` | 500+ | 2000-2020 | 100 MB |

### Manual Download (If API Fails)

1. Go to: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
2. Click "Download"
3. Extract to: `data/kaggle/nifty50-stock-market-data/`
4. Run: `python scripts/download_kaggle_data.py --process-local`

---

## 3. Combined Data Strategy

### Recommended Workflow

```bash
# Step 1: Download FREE sentiment data (HuggingFace)
python scripts/download_hf_news.py --all
# Result: 70K+ labeled articles for sentiment training!

# Step 2: Download FREE historical prices (Kaggle)
python scripts/download_kaggle_data.py
# Result: 20 years of NIFTY 50 data

# Step 3: Download recent prices (yfinance - FREE)
python scripts/collect_prices.py --years 5
# Result: Fill gaps from 2021-present

# Step 4: Use Firecrawl for additional news (500K credits)
python scripts/collect_news.py --target 50000
# Result: More recent news for training

# Step 5: Use Zerodha for minute data
python scripts/collect_minute_data.py
# Result: High-frequency data for strategy
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   PRICE DATA                                                            │
│   ───────────                                                           │
│   Kaggle (2000-2021) ─────┐                                            │
│                           ├──► Merged Historical ──► Training          │
│   yfinance (2021-now) ────┘                                            │
│                                                                         │
│   SENTIMENT DATA                                                        │
│   ──────────────                                                        │
│   HuggingFace (pre-labeled) ──► 70K+ articles ──► Sentiment Model      │
│   Firecrawl (new articles) ────► 50K+ articles ─┘                      │
│                                                                         │
│   MINUTE DATA                                                           │
│   ───────────                                                           │
│   Zerodha Kite ──► 60 days minute ──► Strategy Training                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Cost Savings

### Without FREE Sources:
| Data | Cost |
|------|------|
| Sentiment labeling (10K articles × ₹5) | ₹50,000 |
| Historical data provider | ₹20,000/year |
| **Total** | **₹70,000+** |

### With FREE Sources:
| Data | Cost |
|------|------|
| HuggingFace sentiment | FREE |
| Kaggle historical | FREE |
| yfinance recent | FREE |
| **Total** | **₹0** |

**You save: ₹70,000+ 🎉**

---

## 5. Quick Start Commands

```bash
# Install additional dependencies
pip install datasets kaggle

# Download everything FREE
python scripts/download_hf_news.py --all    # Sentiment data
python scripts/download_kaggle_data.py      # Price history
python scripts/collect_prices.py --years 5  # Recent prices
python scripts/collect_fundamentals.py      # Fundamentals

# Check what you have
python scripts/collect_all_data.py --status
```

---

## 6. Using the Data

### For Sentiment Model Training

```python
import pandas as pd

# Load pre-labeled sentiment data
df = pd.read_parquet("data/processed/sentiment/sentiment_training_data.parquet")

print(f"Total: {len(df):,} labeled articles")
print(df['label'].value_counts())

# Ready for fine-tuning!
# - 70K+ articles
# - Already labeled
# - No manual work needed!
```

### For Price Prediction

```python
import pandas as pd

# Load historical prices
kaggle_df = pd.read_parquet("data/processed/prices/kaggle_nifty50_historical.parquet")
recent_df = pd.read_parquet("data/raw/prices/RELIANCE.parquet")

# Merge for complete history
combined = pd.concat([kaggle_df, recent_df]).drop_duplicates()
print(f"Total history: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
```

---

## Summary

| Task | Script | Cost | Time |
|------|--------|------|------|
| Sentiment data | `download_hf_news.py --all` | FREE | 5 min |
| Historical prices | `download_kaggle_data.py` | FREE | 5 min |
| Recent prices | `collect_prices.py` | FREE | 15 min |
| Fundamentals | `collect_fundamentals.py` | FREE | 15 min |
| **Total FREE data** | | **FREE** | **~40 min** |

After this, you'll have:
- ✅ 70K+ labeled sentiment articles
- ✅ 20+ years of price history
- ✅ Fundamentals for 100 stocks
- ✅ Ready for model training!

Then use your paid resources (Firecrawl, Zerodha) to **supplement** this FREE foundation.
