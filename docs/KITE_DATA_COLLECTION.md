# Kite API Data Collection Guide

Complete guide to fetch historical intraday data from Zerodha Kite API.

---

## 🎯 Overview

Collect 3+ years of 5-minute intraday data for NIFTY 50/100 stocks using Kite Connect API.

**What You'll Get:**
- Historical OHLCV data (5-minute intervals)
- Up to 10+ years of history
- 50-100 liquid stocks
- Ready for ML training

---

## 📋 Prerequisites

### 1. Zerodha Account
- Active Zerodha trading account
- Demat account (optional)

### 2. Kite Connect API Access
- Create app at: https://developers.kite.trade/
- Note: API access may require approval

### 3. Install Dependencies
```bash
pip install kiteconnect pandas pyarrow
```

---

## 🚀 Quick Start

### Step 1: Get API Credentials

1. Go to: https://developers.kite.trade/
2. Login with Zerodha credentials
3. Click "Create new app"
4. Fill details:
   - **App name**: InvestLLM Data Collector
   - **Redirect URL**: http://127.0.0.1
   - **Description**: Historical data collection for ML
5. Note down:
   - **API Key**: (shown after app creation)
   - **API Secret**: (click "Show" to reveal)

---

### Step 2: Generate Access Token

**Easy Method** (Use our helper script):

```bash
python scripts/kite_auth_helper.py
```

Follow the prompts:
1. Enter API Key
2. Enter API Secret
3. Open the login URL in browser
4. Copy request_token from redirect URL
5. Paste request_token

**Output:**
- Access token displayed
- Saved to `.env.kite` file

---

### Step 3: Fetch Historical Data

**Fetch NIFTY 50 (Last 3 years):**

```bash
# Load credentials
source .env.kite

# Fetch data
python scripts/kite_data_collector.py \
    --symbols NIFTY50 \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --interval 5minute \
    --output data/kite_historical/nifty50_2021_2024.parquet
```

**Fetch NIFTY 100 (Last 3 years):**

```bash
python scripts/kite_data_collector.py \
    --symbols NIFTY100 \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --interval 5minute \
    --output data/kite_historical/nifty100_2021_2024.parquet
```

**Custom Symbols:**

```bash
python scripts/kite_data_collector.py \
    --symbols CUSTOM \
    --custom-symbols RELIANCE TCS INFY HDFCBANK ICICIBANK \
    --start-date 2021-01-01 \
    --end-date 2024-12-31 \
    --interval 5minute \
    --output data/kite_historical/custom_stocks.parquet
```

---

## ⏱️ Time & Cost Estimates

### Data Collection Time:

| Symbols | Period | Bars/Symbol | Total Bars | Time (Est) |
|---------|--------|-------------|------------|------------|
| 50 | 1 year | ~18,750 | 937K | ~15 min |
| 50 | 3 years | ~56,250 | 2.8M | ~45 min |
| 100 | 3 years | ~56,250 | 5.6M | ~90 min |

**Rate Limit**: 3 requests/second (built into script)

### API Costs:

Zerodha Kite Connect pricing:
- **Free for testing** (with your own trading account)
- **₹2,000/month** for commercial use
- Historical data API: **Included** (no per-request charges)

---

## 📊 Data Format

### Output Columns:

```
TIME             - Timestamp (datetime)
SYMBOL           - Stock symbol (string)
OPEN_PRICE       - Open price (float)
HIGH_PRICE       - High price (float)
LOW_PRICE        - Low price (float)
CLOSE_PRICE      - Close price (float)
VOLUME           - Volume traded (int)
TOKEN            - Instrument token (int)
EXCHANGE         - Exchange name (string)
```

### Example Data:

```
TIME                SYMBOL   OPEN_PRICE  HIGH_PRICE  LOW_PRICE  CLOSE_PRICE  VOLUME
2021-01-01 09:15:00 RELIANCE 2350.50     2355.75     2348.20    2352.30      125000
2021-01-01 09:20:00 RELIANCE 2352.50     2358.00     2351.80    2357.20      98000
...
```

---

## 🔧 Advanced Options

### Different Intervals:

```bash
# 1-minute data (more granular, larger file)
--interval minute

# 15-minute data (less granular, smaller file)
--interval 15minute

# Daily data (for long-term analysis)
--interval day
```

### BSE Exchange:

```bash
--exchange BSE
```

### Date Range Tips:

**Maximum History:**
- Kite API: ~10 years of intraday data
- Daily data: Unlimited

**Recommended:**
- For ML training: **3-5 years**
- For quick testing: **6 months**
- For production: **5+ years**

---

## 🐛 Troubleshooting

### Error: "Invalid API credentials"
**Solution:**
- Check API Key and Secret are correct
- Access token may have expired (generate new one daily)

### Error: "Rate limit exceeded"
**Solution:**
- Script has built-in rate limiting (3 req/s)
- If still hitting limits, increase `REQUEST_DELAY` in script

### Error: "Symbol not found"
**Solution:**
- Check symbol name (use: `RELIANCE`, not `RELIANCE.NS`)
- Some stocks may not have intraday data

### Error: "Historical data not available"
**Solution:**
- Stock may be newly listed
- Try shorter date range
- Check if symbol is correct

### Access Token Expires Daily
**Solution:**
- Run `kite_auth_helper.py` daily to get new token
- Or implement automatic token refresh (advanced)

---

## 📈 Next Steps After Data Collection

### 1. Feature Engineering:

```bash
python scripts/intraday_feature_engineering.py \
    --input data/kite_historical/nifty100_2021_2024.parquet \
    --output data/intraday_features/train_features.parquet
```

### 2. Train Model:

```bash
# Local training
python scripts/train_intraday_model.py \
    --data data/intraday_features/train_features.parquet \
    --epochs 100

# Or upload to RunPod GPU for faster training
```

### 3. Backtest:

```bash
python scripts/backtest_intraday_model.py \
    --model trained_models/intraday/best_model.ckpt \
    --data data/intraday_features/train_features.parquet \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

---

## 💡 Best Practices

### Data Collection:

1. **Start Small**: Test with 1-2 months first
2. **Incremental**: Fetch year-by-year, save checkpoints
3. **Verify**: Check data completeness after fetching
4. **Backup**: Save raw data before processing

### Storage:

- **Parquet format**: Compressed, efficient
- **Organize by date**: `data/kite_historical/2021_Q1.parquet`
- **Keep raw data**: Don't overwrite original fetches

### API Usage:

- **Respect rate limits**: Don't modify REQUEST_DELAY
- **Token management**: Regenerate daily
- **Error handling**: Script retries automatically
- **Monitor**: Watch console output for errors

---

## 📚 Additional Resources

### Official Documentation:
- Kite Connect API: https://kite.trade/docs/connect/v3/
- Historical Data: https://kite.trade/docs/connect/v3/historical/

### Support:
- Kite API Forum: https://forum.kite.trade/
- Zerodha Support: https://support.zerodha.com/

---

## 🔐 Security Notes

### Protect Your Credentials:

```bash
# NEVER commit credentials to git
echo ".env.kite" >> .gitignore

# Use environment variables
export KITE_API_KEY="your_key"
export KITE_ACCESS_TOKEN="your_token"

# Or use .env file (don't commit!)
```

### API Key Safety:
- Don't share API keys publicly
- Regenerate if compromised
- Use separate keys for dev/prod

---

## ❓ FAQ

**Q: Is Kite API free?**
A: Free for personal use with your trading account. ₹2,000/month for commercial.

**Q: How much data can I fetch?**
A: Up to 10 years of intraday, unlimited daily data.

**Q: Does this work for Futures & Options?**
A: Yes! Just modify symbol format (e.g., "NIFTY24JANFUT")

**Q: Can I run this daily to update data?**
A: Yes! Fetch only new dates, append to existing file.

**Q: What about weekends/holidays?**
A: Script automatically skips non-trading days.

---

**Ready to collect data!** 🚀

Run:
```bash
python scripts/kite_auth_helper.py
```

Then start fetching your 3+ years of historical data!
