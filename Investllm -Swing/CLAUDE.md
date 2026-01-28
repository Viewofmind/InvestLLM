# Using Claude AI with InvestLLM Swing Trading 🤖

> Guide for leveraging Claude AI in the momentum strategy development

---

## 🎯 Project Context

InvestLLM Swing Trading is a **monthly momentum strategy** that:
- Holds top 20 stocks by 12-month momentum
- Rebalances monthly (replace 5 weakest with 5 strongest)
- Uses trailing stops (15%) and volatility filters
- Achieves **23.76% CAGR** with **11.76% alpha over NIFTY**

---

## 📊 Key Files

| File | Purpose |
|------|---------|
| `monthly_momentum_v5.py` | Production strategy with risk controls |
| `monthly_momentum_v4.py` | Base momentum strategy (no stops) |
| `momentum_backtester_v4.py` | 20-day breakout strategy |
| `ensemble_predictor.py` | ML model with temperature scaling |
| `swing_backtester_v3.py` | GPU-accelerated ML backtester |
| `swing_exit_strategy_v3.py` | Dynamic exit logic |

---

## 🚀 Common Tasks

### 1. Backtest a Strategy

**Prompt:**
```
Run the V5 momentum strategy backtest with:
- Starting capital: ₹5,00,000
- Portfolio size: 25 stocks
- Trailing stop: 20%
Show me the CAGR, max drawdown, and Sharpe ratio.
```

**Expected Output:**
```bash
python monthly_momentum_v5.py --capital 500000 --portfolio-size 25 --trailing-stop 0.20
```

---

### 2. Add a New Risk Control

**Prompt:**
```
I want to add a market regime filter to V5 that:
- Checks if NIFTY is above 200-day MA
- Reduces position size by 50% in bear markets
- Goes to 100% cash if NIFTY drops 20% from peak

Modify monthly_momentum_v5.py to include this.
```

---

### 3. Analyze Backtest Results

**Prompt:**
```
Read the backtest results from reports/monthly_momentum_v5/trades_v5.csv
and tell me:
1. Which stocks had the highest win rate?
2. What was the average holding period?
3. How much did trailing stops save us?
```

---

### 4. Compare Strategies

**Prompt:**
```
Compare the performance of:
- V5 (with risk controls)
- V4 (no risk controls)
- V3 (ML LSTM)

Create a table with CAGR, Max DD, Sharpe, Alpha.
Explain why simple momentum beat ML.
```

---

### 5. Debug an Issue

**Prompt:**
```
The backtest shows 0 trades. Here's the error:
[paste error]

Here's the relevant code:
[paste code section]

What's wrong and how do I fix it?
```

---

## 💡 Strategy Development Prompts

### Improve Returns

```
Suggest 5 ways to improve the V5 strategy's CAGR without
significantly increasing max drawdown. Focus on:
- Stock selection filters
- Entry timing
- Position sizing
- Exit optimization
```

### Reduce Risk

```
The max drawdown is -47%. How can we reduce this to -30%
while keeping at least 18% CAGR? Consider:
- Market regime filters
- Sector diversification
- Hedging with NIFTY shorts
```

### Add New Factor

```
I want to add a quality factor (ROE, Debt/Equity) to the
momentum strategy. Show me how to:
1. Download fundamental data
2. Calculate quality score
3. Combine with momentum for ranking
4. Modify the strategy code
```

---

## 🔧 Code Review Prompts

### Review for Bugs

```
Review monthly_momentum_v5.py for:
- Off-by-one errors in date indexing
- Look-ahead bias in backtesting
- Edge cases in position sizing
```

### Optimize Performance

```
The backtest takes 5 minutes to run. Suggest ways to:
- Vectorize calculations
- Reduce memory usage
- Parallelize stock processing
```

### Improve Code Quality

```
Refactor the RiskManager class to:
- Use dataclasses for configuration
- Add type hints
- Include docstrings
- Make it more testable
```

---

## 📈 Analysis Prompts

### Drawdown Analysis

```
Analyze the equity curve from equity_curve_v5.csv:
1. Identify the 3 largest drawdowns
2. What market events caused them?
3. How long did recovery take?
4. Did the risk controls help?
```

### Trade Analysis

```
From trades_v5.csv, calculate:
1. Average winner vs average loser
2. Win rate by month of year
3. Best/worst performing stocks
4. Holding period distribution
```

### Risk Metrics

```
Calculate these risk metrics from the backtest:
- Sortino Ratio
- Calmar Ratio
- Maximum consecutive losses
- Value at Risk (95%)
```

---

## 🧪 Testing Prompts

### Write Unit Tests

```
Write pytest tests for the RiskManager class that verify:
- Trailing stop triggers at correct level
- Volatility filter reduces position size
- Max loss is enforced
```

### Validate Backtest

```
Help me validate the backtest is realistic:
1. Check for look-ahead bias
2. Verify transaction cost modeling
3. Test with different starting dates
4. Compare with manual calculation
```

---

## 📚 Learning Prompts

### Explain Concepts

```
Explain why simple 12-month momentum works better than
complex ML models for stock selection. Include:
- Academic research
- Market efficiency arguments
- Practical considerations
```

### Strategy Design

```
I want to create a mean-reversion strategy to complement
momentum. Explain:
- How mean reversion differs from momentum
- When to use each
- How to combine them
```

---

## ⚠️ Important Guidelines

### DO ✅
- Share backtest results and code
- Ask for optimization suggestions
- Request code reviews
- Discuss strategy ideas

### DON'T ❌
- Share API keys or credentials
- Ask to bypass risk controls
- Request real-time trading without testing
- Ignore drawdown warnings

---

## 🎯 Key Metrics to Track

| Metric | V5 Result | Target |
|--------|-----------|--------|
| CAGR | 23.76% | > 20% |
| Max Drawdown | -47.30% | < 30% |
| Sharpe Ratio | 1.22 | > 1.0 |
| Alpha vs NIFTY | +11.76% | > 8% |
| Win Rate | 49.9% | > 50% |
| Profit Factor | 3.03 | > 2.0 |

---

## 🔄 Iteration Workflow

1. **Hypothesis**: "Adding X will improve Y"
2. **Implement**: Modify code with Claude's help
3. **Backtest**: Run on historical data
4. **Analyze**: Review results with Claude
5. **Refine**: Adjust parameters
6. **Document**: Update README/ROADMAP

---

## 📞 Quick Reference

### Run V5 Backtest
```bash
python monthly_momentum_v5.py --features swing_features/ --capital 100000
```

### Compare with V4
```bash
python monthly_momentum_v5.py --no-trailing-stop --no-vol-filter
```

### Check Results
```bash
cat reports/monthly_momentum_v5/trades_v5.csv | head -20
```

---

*Claude AI Guide for InvestLLM Swing Trading*
*Version 5.0 - January 2026*
