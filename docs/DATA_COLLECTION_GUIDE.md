# Complete Data Collection Guide

## Your Resources

| Resource | Available | Purpose |
|----------|-----------|---------|
| **Zerodha Kite API** | ✅ Active | Real-time + minute data |
| **Firecrawl** | 500,000 credits | News scraping |
| **yfinance** | Unlimited (FREE) | 20-year EOD data |

---

## Quick Start Summary

```bash
# Step 1: Collect 20 years price data (FREE)
python scripts/collect_prices.py

# Step 2: Collect fundamentals (FREE)
python scripts/collect_fundamentals.py

# Step 3: Collect minute data (Zerodha - ₹2K/month)
python scripts/collect_minute_data.py

# Step 4: Collect news (Firecrawl - 500K credits)
python scripts/collect_news.py --target 100000
```

---

## Detailed Guide

### Phase 1: Price Data (FREE - yfinance)

#### What You Get:
- 20 years of daily OHLCV data
- 100 stocks (NIFTY 100)
- 10 indices
- ~500,000 rows of data

#### How to Run:

```bash
# Navigate to project
cd InvestLLM

# Activate virtual environment
source venv/bin/activate

# Run price collection
python scripts/collect_prices.py --years 20
```

#### Expected Output:
```
📈 Collecting 100 stocks...
Downloading [====================] 100% (100/100)

✓ RELIANCE (5,024 rows)
✓ TCS (5,018 rows)
...

Collection Complete!
  Stocks: 98/100
  Indices: 10/10
  Failed: 2
  Data saved to: data/raw/prices/
```

#### Time: ~15-20 minutes
#### Cost: FREE

#### Data Location:
```
data/
├── raw/
│   ├── prices/
│   │   ├── RELIANCE.parquet
│   │   ├── TCS.parquet
│   │   └── ... (100 files)
│   └── indices/
│       ├── NIFTY50.parquet
│       └── ... (10 files)
└── price_collection_report.csv
```

---

### Phase 2: Fundamental Data (FREE - yfinance)

#### What You Get:
- Current key ratios (PE, PB, ROE, etc.)
- Financial statements (Income, Balance, Cash Flow)
- Analyst recommendations
- Company info

#### How to Run:

```bash
python scripts/collect_fundamentals.py
```

#### Expected Output:
```
📊 Collecting fundamentals for 100 stocks

Collecting [====================] 100%

Collection Complete!
  Saved to: data/fundamentals/
```

#### Time: ~10-15 minutes
#### Cost: FREE

#### Data Location:
```
data/
└── fundamentals/
    ├── fundamentals.parquet    # All ratios
    ├── fundamentals.csv        # CSV version
    └── statements/
        ├── RELIANCE/
        │   ├── income_statement.parquet
        │   ├── balance_sheet.parquet
        │   └── cashflow.parquet
        └── ... (100 folders)
```

---

### Phase 3: Minute Data (Zerodha Kite - ₹2K/month)

#### What You Get:
- 60 days of minute-level data
- 100 days of 5-minute data
- 2000 days (~5 years) of daily data
- Real-time quotes

#### Prerequisites:

1. **Get Zerodha Kite API Access:**
   - Go to https://kite.trade/
   - Subscribe to Kite Connect (₹2000/month)
   - Get API Key and API Secret

2. **Set Environment Variables:**
   ```bash
   # Add to .env file
   ZERODHA_API_KEY=your_api_key
   ZERODHA_API_SECRET=your_api_secret
   ```

#### How to Run:

```bash
# First run - authentication required
python scripts/collect_minute_data.py

# You'll see:
# Step 1: Open this URL in browser:
# https://kite.trade/connect/login?api_key=XXX
#
# Step 2: Log in with your Zerodha credentials
# Step 3: Copy the 'request_token' from the redirect URL
```

#### Authentication Flow:

```
1. Run script → Get login URL
2. Open URL in browser → Login with Zerodha
3. Redirected to your app URL with request_token
4. Enter request_token → Get access_token
5. Save access_token in .env for future use
```

#### Collection Options:

```bash
# Minute data (60 days max)
python scripts/collect_minute_data.py --interval minute --days 60

# 5-minute data (100 days max)
python scripts/collect_minute_data.py --interval 5minute --days 100

# Daily data (2000 days max - ~5.5 years)
python scripts/collect_minute_data.py --interval day --days 2000
```

#### Time: ~30-60 minutes (depending on stocks)
#### Cost: ₹2000/month

---

### Phase 4: News Collection (Firecrawl - 500K Credits)

#### Credit Strategy:

| Allocation | Credits | Expected Articles |
|------------|---------|-------------------|
| Moneycontrol | 150,000 | ~40,000 |
| Economic Times | 120,000 | ~35,000 |
| Business Standard | 80,000 | ~20,000 |
| LiveMint | 80,000 | ~20,000 |
| NDTV Profit | 70,000 | ~15,000 |
| **Total** | **500,000** | **~130,000** |

#### Prerequisites:

1. **Set Firecrawl API Key:**
   ```bash
   # Add to .env file
   FIRECRAWL_API_KEY=your_firecrawl_key
   ```

#### How to Run:

```bash
# Collect from all sources
python scripts/collect_news.py --target 100000

# Or collect from specific source
python scripts/collect_news.py --source moneycontrol --target 30000
```

#### Progress Monitoring:

```bash
# Check collection status anytime
python scripts/collect_all_data.py --status
```

#### Expected Output:
```
📰 Firecrawl News Collection
Target: 100,000 articles | Credits: 500,000

Crawling Moneycontrol
  Discovering articles in /news/business/stocks/...
  Found 8,234 articles from Moneycontrol

Crawling Economic Times
  Discovering articles in /markets/stocks/news...
  ...

Collection Complete!
┌─────────────────┬───────────┐
│ Metric          │ Value     │
├─────────────────┼───────────┤
│ Total Articles  │ 103,456   │
│ Total Size      │ 2.3 GB    │
│ Credits Used    │ 423,891   │
│ Remaining       │ 76,109    │
└─────────────────┴───────────┘
```

#### Time: 2-4 hours
#### Cost: ~400K-450K credits

#### Data Location:
```
data/
└── news/
    ├── articles_moneycontrol_20240126_101530.json
    ├── articles_economic_times_20240126_112045.json
    └── collected_urls.txt
```

---

## Data Summary

After completing all phases:

| Data Type | Records | Size | Time |
|-----------|---------|------|------|
| Price (EOD) | ~500K rows | ~50 MB | 15 min |
| Price (Minute) | ~2M rows | ~500 MB | 1 hr |
| Fundamentals | 100 stocks | ~10 MB | 15 min |
| Financial Statements | 100 × 6 | ~50 MB | included |
| News Articles | ~100K | ~2 GB | 3 hrs |
| **Total** | | **~2.6 GB** | **~5 hrs** |

---

## Troubleshooting

### yfinance Issues

```bash
# Rate limited?
# The script has built-in delays, but if you still get errors:
python scripts/collect_prices.py --years 20
# Wait 1 hour and retry failed symbols
```

### Zerodha Authentication Failed

```
# Common issues:
1. request_token expires in 3 minutes - be quick!
2. access_token expires daily - regenerate each day
3. Check API subscription is active at kite.trade
```

### Firecrawl Credit Issues

```bash
# Check remaining credits
# Go to https://firecrawl.dev/dashboard

# If running low, reduce target:
python scripts/collect_news.py --target 50000 --credits 200000
```

---

## Automation (Optional)

### Daily Price Updates

Create `scripts/daily_update.sh`:
```bash
#!/bin/bash
cd /path/to/InvestLLM
source venv/bin/activate

# Update prices
python -c "
from investllm.data.collectors.price_collector import PriceDataCollector
collector = PriceDataCollector()
collector.update_daily()
"

# Log
echo "Daily update completed at $(date)" >> logs/daily_update.log
```

Add to crontab:
```bash
# Run at 6 PM IST (after market closes)
30 18 * * 1-5 /path/to/scripts/daily_update.sh
```

---

## Next Steps

After data collection:

1. **Validate Data:**
   ```bash
   python scripts/collect_all_data.py --status
   ```

2. **Start Labeling News (for sentiment model):**
   - Export 2000 random articles
   - Label as positive/negative/neutral
   - Save to `data/labeled/sentiment_labels.csv`

3. **Begin Phase 2: Sentiment Model Training**
   - Fine-tune Mistral 7B
   - Target: 75%+ accuracy

---

## Cost Summary

| Item | One-time | Monthly |
|------|----------|---------|
| yfinance | FREE | FREE |
| Firecrawl (500K) | ₹0 (existing) | - |
| Zerodha Kite | - | ₹2,000 |
| **Total Phase 1** | **₹0** | **₹2,000** |

You have everything you need to complete Phase 1! 🚀
