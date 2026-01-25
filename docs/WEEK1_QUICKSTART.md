# Week 1 Quick Start Guide

## Day 1: Environment Setup

### 1. Install Prerequisites

```bash
# Python 3.11+
python --version  # Should be 3.11+

# Docker (for databases)
docker --version
docker-compose --version
```

### 2. Clone and Setup Project

```bash
cd investllm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env

# Edit .env and set:
# - POSTGRES_PASSWORD (choose a secure password)
# - HF_TOKEN (get from huggingface.co)
# - FIRECRAWL_API_KEY (optional, get from firecrawl.dev)
```

### 4. Start Infrastructure

```bash
# Start PostgreSQL, Redis, Qdrant, MLflow
docker-compose up -d

# Check services are running
docker-compose ps

# Expected output:
# investllm-postgres   running   0.0.0.0:5432->5432/tcp
# investllm-redis      running   0.0.0.0:6379->6379/tcp
# investllm-qdrant     running   0.0.0.0:6333->6333/tcp
# investllm-mlflow     running   0.0.0.0:5000->5000/tcp
```

### 5. Initialize Database

```bash
python scripts/init_db.py

# Expected output:
# ✅ Tables created
# ✅ Stock data seeded
# ✅ Database initialization complete!
```

### 6. Verify Setup

```bash
# Check database
python scripts/init_db.py --check

# Access MLflow UI
open http://localhost:5000

# Access Qdrant UI
open http://localhost:6333/dashboard
```

---

## Day 2-3: Data Collection Start

### Collect Price Data

```bash
# Test with single stock
python -c "
from investllm.data.collectors.price_collector import PriceCollector
collector = PriceCollector()
df = collector.get_stock_history('RELIANCE', years=5)
print(f'Collected {len(df)} rows for RELIANCE')
print(df.head())
"

# Collect all NIFTY 50 (takes ~10 mins)
python -c "
from investllm.data.collectors.price_collector import PriceCollector
from investllm.config import NIFTY_50
collector = PriceCollector()
results = collector.collect_all_historical(symbols=NIFTY_50, years=10)
print(f'Collected data for {len(results)} stocks')
"
```

### Collect Fundamentals

```bash
python -c "
from investllm.data.collectors.fundamental_collector import FundamentalCollector
collector = FundamentalCollector()
data = collector.get_fundamentals('RELIANCE')
print(data)
"
```

---

## Day 4-5: News Collection Setup

### Test News Scraping

```bash
python -c "
import asyncio
from investllm.data.collectors.news_collector import NewsCollector

async def test():
    collector = NewsCollector()
    articles = await collector.collect_recent(pages=2)
    print(f'Collected {len(articles)} articles')
    for a in articles[:3]:
        print(f'  - {a.title[:50]}...')

asyncio.run(test())
"
```

---

## Troubleshooting

### PostgreSQL Connection Failed
```bash
# Check if container is running
docker ps | grep postgres

# Check logs
docker logs investllm-postgres

# Restart
docker-compose restart postgres
```

### yfinance Rate Limited
```bash
# Add delay between requests
# Edit investllm/data/collectors/price_collector.py
# Increase time.sleep() values
```

### Memory Issues
```bash
# Reduce batch size
# Process stocks in smaller batches
```

---

## Week 1 Checklist

- [ ] Python 3.11+ installed
- [ ] Docker running
- [ ] PostgreSQL + TimescaleDB running
- [ ] Redis running
- [ ] Qdrant running
- [ ] MLflow accessible at localhost:5000
- [ ] Database initialized with stock master data
- [ ] Successfully collected RELIANCE price data
- [ ] Successfully collected RELIANCE fundamentals
- [ ] Successfully scraped test news articles

---

## Next: Week 2

Once Week 1 is complete, move to full data collection:
1. Collect 20 years price data for all 100 stocks
2. Collect fundamentals for all stocks
3. Start building news corpus
4. Set up daily update automation
