# InvestLLM Swing Trading - Roadmap 🗺️

> Future development plans for the momentum strategy

---

## 🎯 Current State (V5)

**Achieved:**
- 23.76% CAGR with 11.76% alpha over NIFTY
- Risk controls: 15% trailing stop, 20% max loss
- Volatility-adjusted position sizing
- 18.8 years backtested (2007-2025)

---

## 📅 Phase 1: Risk Reduction (Q1 2026)

### Goal: Reduce max drawdown from -47% to <30%

| Feature | Priority | Description |
|---------|----------|-------------|
| **Market Regime Filter** | High | Avoid entering new positions in bear markets |
| **Sector Diversification** | High | Limit sector concentration to 30% |
| **Correlation Filter** | Medium | Reduce portfolio correlation |
| **ATR-Based Stops** | Medium | Dynamic stops based on stock volatility |

### Technical Implementation

```python
# Market Regime Detection
class MarketRegimeFilter:
    def is_bull_market(self, nifty_data):
        # NIFTY above 200-day MA
        ma_200 = nifty_data['close'].rolling(200).mean()
        return nifty_data['close'].iloc[-1] > ma_200.iloc[-1]

# Sector Limits
max_sector_weight = 0.30  # Max 30% in any sector
```

### Expected Outcome
- Max Drawdown: -47% → -30%
- CAGR: 23.76% → ~20% (trade-off for safety)
- Calmar Ratio: 0.50 → 0.67

---

## 📅 Phase 2: Alpha Enhancement (Q2 2026)

### Goal: Increase CAGR while maintaining risk profile

| Feature | Priority | Description |
|---------|----------|-------------|
| **Dual Momentum** | High | Absolute + Relative momentum |
| **Quality Filter** | High | ROE > 15%, Debt/Equity < 1 |
| **Earnings Momentum** | Medium | EPS growth acceleration |
| **Sector Rotation** | Medium | Overweight best sectors |

### Dual Momentum Strategy

```python
def dual_momentum_signal(stock_return, nifty_return, tbill_rate):
    # Relative momentum: stock vs index
    relative = stock_return > nifty_return

    # Absolute momentum: stock vs risk-free
    absolute = stock_return > tbill_rate

    # Both must be positive
    return relative and absolute
```

### Expected Outcome
- CAGR: 20% → 25%+
- Win Rate: 50% → 55%
- Profit Factor: 3.0 → 3.5

---

## 📅 Phase 3: Live Trading (Q3 2026)

### Goal: Production deployment with broker integration

| Feature | Priority | Description |
|---------|----------|-------------|
| **Zerodha Integration** | High | Kite Connect API |
| **Real-Time Monitoring** | High | Position tracking dashboard |
| **Alert System** | High | Telegram/Email notifications |
| **Order Management** | Medium | Smart order routing |

### Broker Integration

```python
# Zerodha Kite Integration
from kiteconnect import KiteConnect

class LiveTrader:
    def __init__(self, api_key, access_token):
        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)

    def execute_rebalance(self, sells, buys):
        # Sell first
        for symbol, qty in sells:
            self.kite.place_order(
                tradingsymbol=symbol,
                exchange="NSE",
                transaction_type="SELL",
                quantity=qty,
                order_type="MARKET"
            )

        # Then buy
        for symbol, qty in buys:
            self.kite.place_order(...)
```

### Deployment Checklist
- [ ] Paper trading for 1 month
- [ ] API integration tested
- [ ] Alert system configured
- [ ] Risk limits set
- [ ] Emergency stop implemented

---

## 📅 Phase 4: Advanced Features (Q4 2026)

### Goal: Sophisticated strategy enhancements

| Feature | Priority | Description |
|---------|----------|-------------|
| **Options Overlay** | Medium | Covered calls on holdings |
| **Factor Timing** | Medium | Rotate between value/momentum |
| **Volatility Targeting** | Low | Maintain constant portfolio vol |
| **Multi-Asset** | Low | Include bonds/gold allocation |

### Options Overlay Example

```python
# Covered Call Strategy
def sell_covered_call(holding, strike_pct=1.05, expiry_days=30):
    """
    Sell OTM call on existing holdings
    - Strike: 5% above current price
    - Expiry: Monthly (30 days)
    - Expected income: 1-2% per month
    """
    strike = holding.current_price * strike_pct
    premium = get_call_premium(holding.symbol, strike, expiry_days)
    return premium
```

---

## 📅 Phase 5: ML Integration (2027)

### Goal: Use ML for enhancement, not core signals

| Feature | Priority | Description |
|---------|----------|-------------|
| **Regime Detection** | High | Classify bull/bear/sideways |
| **Risk Forecasting** | High | Predict future volatility |
| **Anomaly Detection** | Medium | Identify unusual patterns |
| **Sentiment Analysis** | Low | News-based risk adjustment |

### ML Use Cases (Correct Approach)

```python
# GOOD: ML for risk management
class MLRiskManager:
    def predict_volatility(self, features):
        # Forecast 5-day realized volatility
        return self.vol_model.predict(features)

    def adjust_position_size(self, base_size, predicted_vol):
        # Reduce size when high vol predicted
        return base_size * (0.20 / predicted_vol)

# BAD: ML for signal generation (doesn't work)
# def predict_next_day_return():
#     return model.predict(...)  # Too noisy!
```

---

## 📊 Feature Comparison Matrix

| Version | CAGR | Max DD | Sharpe | Key Feature |
|---------|------|--------|--------|-------------|
| V3 (ML) | 5.4% | -15% | 0.41 | LSTM ensemble |
| V4 | 20.2% | -64% | 0.88 | 12-month momentum |
| **V5** | **23.8%** | -47% | 1.22 | + Trailing stops |
| V6 (planned) | ~25% | -30% | 1.5+ | + Regime filter |
| V7 (planned) | ~28% | -25% | 1.8+ | + Dual momentum |

---

## 🔧 Technical Debt & Improvements

### Code Quality
- [ ] Add unit tests for all strategies
- [ ] Type hints throughout codebase
- [ ] Logging framework
- [ ] Configuration management (YAML)

### Performance
- [ ] Vectorized backtesting (10x faster)
- [ ] Parallel parameter optimization
- [ ] Memory-efficient data loading

### Documentation
- [ ] API documentation
- [ ] Strategy whitepaper
- [ ] Video tutorials

---

## 💡 Research Ideas

### To Investigate
1. **Mean Reversion Combo**: Momentum + short-term mean reversion
2. **Volatility Breakout**: Enter on volatility expansion
3. **Calendar Effects**: Month-end/start patterns
4. **Liquidity Premium**: Focus on mid-cap names
5. **Pairs Trading**: Long/short within sectors

### Academic Papers to Study
- Asness (2014): "Fact, Fiction, and Momentum"
- Frazzini (2014): "Betting Against Beta"
- Novy-Marx (2012): "Is Momentum Really Momentum?"

---

## 📈 Success Metrics

### Q1 2026 Targets
- [ ] Max Drawdown < 35%
- [ ] Live paper trading started
- [ ] Sector diversification implemented

### Q2 2026 Targets
- [ ] Dual momentum tested
- [ ] Quality filter added
- [ ] CAGR maintained > 20%

### Q3 2026 Targets
- [ ] Zerodha integration complete
- [ ] First live trade executed
- [ ] Alert system operational

### Full Year 2026
- [ ] ₹5L capital deployed
- [ ] 15%+ actual returns
- [ ] < 20% max drawdown experienced

---

## 🤝 Contributing

### Priority Areas
1. Risk management improvements
2. Broker API integration
3. Dashboard development
4. Strategy backtesting framework

### How to Contribute
1. Fork the repository
2. Create feature branch
3. Write tests
4. Submit PR with description

---

## 📞 Feedback

Share ideas and feedback:
- GitHub Issues
- Strategy discussion
- Bug reports

---

*InvestLLM Swing Trading Roadmap*
*Last Updated: January 2026*
*Version: 5.0*
