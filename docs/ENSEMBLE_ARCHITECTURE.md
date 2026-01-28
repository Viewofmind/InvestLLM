# InvestLLM Ensemble AI Architecture

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      INVESTLLM ENSEMBLE SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
│  │   PRICE      │  │  SENTIMENT   │  │ FUNDAMENTAL  │                  │
│  │   MODEL      │  │    MODEL     │  │    MODEL     │                  │
│  │   (LSTM)     │  │  (FinBERT)   │  │  (XGBoost)   │                  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
│         │                 │                 │                          │
│         │    Price        │   Sentiment     │   Quality                │
│         │    Signal       │   Score         │   Score                  │
│         │   (-1 to +1)    │  (-1 to +1)     │  (0 to 1)                │
│         │                 │                 │                          │
│         └────────────────┬┴─────────────────┘                          │
│                          │                                              │
│                          ▼                                              │
│                 ┌─────────────────┐                                     │
│                 │  META-LEARNER   │                                     │
│                 │  (Combines All) │                                     │
│                 └────────┬────────┘                                     │
│                          │                                              │
│                          ▼                                              │
│                 ┌─────────────────┐                                     │
│                 │  SMART EXIT     │                                     │
│                 │  RISK MANAGER   │                                     │
│                 └────────┬────────┘                                     │
│                          │                                              │
│                          ▼                                              │
│                    FINAL DECISION                                       │
│                  (BUY / SELL / HOLD)                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Price Model (LSTM) ✅ DONE

**Status:** Complete
**Location:** `models/price_prediction/`

| Attribute | Value |
|-----------|-------|
| Architecture | LSTM (256 hidden, 3 layers) |
| Input | 21 technical features |
| Output | Log return prediction |
| Training | GPU (RunPod RTX 4090) |
| Accuracy | 51.6% |

---

## Component 2: Sentiment Model (FinBERT) 🔄 IN PROGRESS

**Purpose:** Analyze financial news and extract sentiment for each stock.

### Model Options

| Model | Parameters | Speed | Accuracy | GPU Required |
|-------|------------|-------|----------|--------------|
| DistilBERT | 66M | Fast | Good | Optional |
| FinBERT | 110M | Medium | Best for Finance | Yes |
| TinyBERT | 14M | Very Fast | Moderate | No |

**Recommended:** FinBERT (pre-trained on financial text)

### Training Data Sources

1. **FinGPT Sentiment Dataset** (already downloaded)
   - 150K+ labeled financial texts
   - Labels: positive, negative, neutral

2. **Indian Financial News** (to collect)
   - Economic Times, Moneycontrol, Business Standard
   - Map to stock tickers

### Output Format

```python
{
    "ticker": "RELIANCE",
    "date": "2024-01-15",
    "sentiment_score": 0.72,      # -1 to +1
    "confidence": 0.85,            # 0 to 1
    "news_count": 5,               # Number of news analyzed
    "top_keywords": ["profit", "growth", "expansion"]
}
```

---

## Component 3: Fundamental Model (XGBoost) 📊 PENDING

**Purpose:** Score stocks based on fundamental health.

### Input Features

| Category | Features |
|----------|----------|
| Valuation | P/E, P/B, EV/EBITDA, PEG |
| Profitability | ROE, ROA, ROCE, Profit Margin |
| Growth | Revenue Growth, EPS Growth, Book Value Growth |
| Debt | Debt/Equity, Interest Coverage, Current Ratio |
| Efficiency | Asset Turnover, Inventory Turnover |

### Output

```python
{
    "ticker": "RELIANCE",
    "quality_score": 0.78,        # 0 to 1 (higher = better)
    "value_score": 0.65,          # Is it undervalued?
    "growth_score": 0.82,         # Growth potential
    "safety_score": 0.71          # Financial stability
}
```

---

## Component 4: Meta-Learner 🧠 PENDING

**Purpose:** Combine all signals into final trading decision.

### Approach 1: Weighted Average (Simple)

```python
final_signal = (
    w1 * price_signal +      # Weight: 0.4
    w2 * sentiment_score +   # Weight: 0.3
    w3 * quality_score       # Weight: 0.3
)

decision = "BUY" if final_signal > threshold else "HOLD"
```

### Approach 2: Gradient Boosting (Advanced)

Train XGBoost on:
- Price model prediction
- Sentiment score
- Fundamental scores
- Target: Actual future returns

### Approach 3: Neural Network (Most Advanced)

```python
class MetaLearner(nn.Module):
    def __init__(self):
        self.fc = nn.Sequential(
            nn.Linear(5, 32),  # 5 input signals
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()  # Output: -1 to +1
        )
```

---

## Training Pipeline

### Phase 1: Individual Model Training

```
1. Price Model      → RunPod GPU (Done ✅)
2. Sentiment Model  → RunPod GPU (FinBERT fine-tuning)
3. Fundamental Model → Local CPU (XGBoost)
```

### Phase 2: Meta-Learner Training

```
1. Generate predictions from all 3 models on training data
2. Align predictions by date and ticker
3. Train meta-learner on combined features
4. Validate on held-out period
```

### Phase 3: Backtest & Optimize

```
1. Run ensemble on test period
2. Apply Smart Exit risk management
3. Optimize weights/thresholds
4. Paper trade validation
```

---

## Data Flow (Daily)

```
Morning (Pre-Market):
├── Collect overnight news → Sentiment Model → Daily sentiment score
├── Update fundamental data (quarterly) → Fundamental Model → Quality score
└── Generate combined signal for each stock

Market Hours:
├── Monitor price action
├── Update intraday signals (optional)
└── Smart Exit checks positions

End of Day:
├── Record actual returns
├── Update training data
└── Retrain models (weekly/monthly)
```

---

## Expected Performance

| Metric | Price Only | Ensemble (Expected) |
|--------|------------|---------------------|
| Win Rate | 51-52% | 55-60% |
| Sharpe | 0.5-1.0 | 1.5-2.0 |
| Max DD | -30% | -15% |
| CAGR | 12-15% | 18-25% |

---

## Implementation Order

1. ✅ Price Model (LSTM) - DONE
2. 🔄 Sentiment Model (FinBERT on RunPod) - NEXT
3. 📊 Fundamental Model (XGBoost)
4. 🧠 Meta-Learner
5. 🔧 Integration & Backtesting
6. 📱 Live Trading System

---

## Cost Estimate

| Component | Platform | Cost |
|-----------|----------|------|
| Sentiment Training | RunPod (A100) | $5-10 |
| Fundamental Model | Local | Free |
| Meta-Learner | Local | Free |
| Backtesting | Local | Free |
| **Total** | | **$5-10** |

---

## Files to Create

```
investllm/
├── models/
│   ├── price_prediction/     ✅ Done
│   ├── sentiment/            🔄 Create
│   │   ├── finbert_model.py
│   │   └── sentiment_scorer.py
│   ├── fundamental/          📊 Create
│   │   └── quality_scorer.py
│   └── ensemble/             🧠 Create
│       └── meta_learner.py
├── cloud/
│   ├── train_runpod.py       ✅ Done
│   └── train_sentiment.py    🔄 Create
└── scripts/
    └── ensemble_backtester.py 🧠 Create
```
