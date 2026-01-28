# InvestLLM - Swing Trading Strategy V5 📊

> **Monthly Momentum Strategy with Risk Controls**
>
> **23.76% CAGR | 11.76% Alpha over NIFTY | Sharpe 1.22**

---

## 🎯 Strategy Overview

This is a **production-ready momentum strategy** that beats NIFTY 50 by 11.76% annually over 18.8 years of backtesting.

### Key Results (2007-2025)

| Metric | V5 (With Risk) | V4 (No Risk) | NIFTY 50 |
|--------|---------------|--------------|----------|
| **CAGR** | **23.76%** | 20.19% | ~12% |
| **Max Drawdown** | **-47.30%** | -63.82% | ~55% |
| **Sharpe Ratio** | **1.22** | 0.88 | ~0.5 |
| **Profit Factor** | **3.03** | 0.96 | - |
| **Alpha** | **+11.76%** | +8.19% | 0% |
| **Final Equity** | **₹55.2L** | ₹31.8L | ~₹10L |

Starting with ₹1,00,000 in 2007 → ₹55,21,722 in 2025

---

## 📦 Strategy Components

### Core Strategy (monthly_momentum_v5.py)

1. **Stock Selection**: Top 20 stocks by 12-month momentum
2. **Rebalancing**: Monthly turnover of 5 stocks
3. **Position Sizing**: Equal weight with volatility adjustment

### Risk Controls

| Control | Setting | Purpose |
|---------|---------|---------|
| Trailing Stop | 15% | Lock in profits from peak |
| Max Position Loss | 20% | Limit per-trade losses |
| Volatility Filter | On | Reduce exposure in high-vol |
| Daily Monitoring | On | Check stops every trading day |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pandas, numpy
- Price data in parquet format

### Run Backtest

```bash
# With risk controls (recommended)
python monthly_momentum_v5.py --features swing_features/ --capital 100000

# Without risk controls (comparison)
python monthly_momentum_v5.py --features swing_features/ --no-trailing-stop --no-vol-filter
```

### Output
```
CAGR: 23.76%
Max Drawdown: -47.30%
Sharpe Ratio: 1.22
Final Equity: ₹5,521,722
```

---

## 📁 File Structure

```
Investllm -Swing/
├── monthly_momentum_v5.py      # V5 Strategy with risk controls
├── monthly_momentum_v4.py      # Base momentum strategy
├── momentum_backtester_v4.py   # 20-day breakout strategy
├── ensemble_predictor.py       # ML temperature scaling
├── swing_backtester_v3.py      # GPU-accelerated backtester
├── swing_exit_strategy_v3.py   # Dynamic exit logic
├── gpu_package_v3/             # GPU training scripts
│   ├── train_ensemble.py
│   ├── run_gpu_v3.sh
│   └── README_V3.txt
└── reports/                    # Backtest results
```

---

## 📊 Strategy Comparison

### All Strategies Tested

| Strategy | CAGR | Max DD | Sharpe | Alpha |
|----------|------|--------|--------|-------|
| **V5 Monthly Momentum** | **23.76%** | -47% | 1.22 | +11.76% |
| V4 Monthly Momentum | 20.19% | -64% | 0.88 | +8.19% |
| V4 20-Day Breakout | 5.67% | -42% | 0.38 | -6.33% |
| V3 ML (LSTM Ensemble) | 5.42% | -15% | 0.41 | -6.58% |

### Key Insight
> Simple momentum rules (12-month returns) beat complex ML models for stock selection.
> Risk controls (trailing stops) IMPROVE returns by protecting capital for compounding.

---

## 🔧 Customization

### Adjust Trailing Stop
```bash
python monthly_momentum_v5.py --trailing-stop 0.20  # 20% trailing stop
```

### Change Portfolio Size
```bash
python monthly_momentum_v5.py --portfolio-size 30 --turnover 8
```

### Disable Risk Controls
```bash
python monthly_momentum_v5.py --no-trailing-stop --no-vol-filter
```

---

## 📈 Historical Performance

### Yearly Equity Progression
```
2007: ₹100,000 (start)
2008: ₹173,868 (survived financial crisis)
2010: ₹241,827
2015: ₹893,220
2020: ₹1,508,892 (survived COVID crash)
2022: ₹3,537,287
2025: ₹5,521,722 (55x return)
```

### Drawdown Analysis
- **2008 Crisis**: V5 dropped to ₹127k (27%) vs V4's ₹104k (45%)
- **2020 COVID**: V5 recovered faster due to trailing stops
- **Risk controls saved ~₹20L** in avoided losses over 18 years

---

## 🧪 ML Strategy (V3) - Why It Failed

We also tested an LSTM ensemble model with:
- 3 models trained with different seeds
- Temperature scaling for confidence calibration
- 54% directional accuracy

**Result**: 5.42% CAGR (below NIFTY's 12%)

**Why**:
- Short-term price movements are mostly noise
- Transaction costs eat into small edges
- Momentum anomaly (months) > ML prediction (days)

---

## 📋 Exit Reason Analysis

| Exit Reason | Count | Avg Return |
|-------------|-------|------------|
| TRAILING_STOP_15% | 661 | Varies |
| MONTHLY_REBALANCE | 105 | Varies |
| MAX_LOSS_20% | 49 | -20% |

Trailing stops accounted for 81% of all exits - actively managing risk.

---

## 🎯 Production Deployment

### Paper Trading Checklist
- [ ] Run backtest on latest data
- [ ] Verify signal generation
- [ ] Test order execution logic
- [ ] Monitor for 1 month

### Live Trading Setup
1. Connect to broker API (Zerodha/Angel)
2. Schedule monthly rebalancing (last trading day)
3. Daily trailing stop monitoring (3:25 PM check)
4. Position sizing based on available capital

---

## ⚠️ Risk Warnings

1. **Past performance doesn't guarantee future results**
2. **-47% max drawdown is significant** - size positions accordingly
3. **Requires 20+ stocks for diversification**
4. **Monthly rebalancing = lower turnover but delayed reaction**

---

## 📚 References

- Jegadeesh & Titman (1993) - "Returns to Buying Winners"
- Moskowitz & Grinblatt (1999) - "Industry Momentum"
- Asness et al. (2013) - "Value and Momentum Everywhere"

---

## 🏆 Summary

| Metric | Value |
|--------|-------|
| Strategy | Monthly Momentum + Risk Controls |
| CAGR | 23.76% |
| Alpha vs NIFTY | +11.76% |
| Max Drawdown | -47.30% |
| Sharpe Ratio | 1.22 |
| Profit Factor | 3.03 |
| Backtest Period | 18.8 years (2007-2025) |
| Total Return | 5,421% |

---

*InvestLLM Swing Trading V5*
*Version 5.0 - January 2026*
*Monthly Momentum with Trailing Stops*
