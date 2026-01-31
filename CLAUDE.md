# InvestLLM - AI Assistant Guide

This document provides essential context for AI assistants working on the InvestLLM codebase.

## Project Overview

InvestLLM is a proprietary AI ensemble system for Indian stock market analysis and algorithmic trading. It combines multiple deep learning models to generate trading signals for NIFTY 100 stocks.

**Key Results:**
- 73.31% average returns across 98 stocks
- 1.38 Sharpe ratio, 62.8% win rate
- 99% sentiment classification accuracy
- 56.12% directional accuracy on intraday predictions

## Technology Stack

| Category | Technologies |
|----------|--------------|
| Language | Python 3.11+ |
| Web Framework | FastAPI, Uvicorn |
| Database | PostgreSQL + TimescaleDB |
| Vector DB | Qdrant |
| Cache | Redis |
| ORM | SQLAlchemy |
| ML | PyTorch 2.2.0, HuggingFace Transformers |
| Frontend | React, TypeScript, Tailwind CSS, Vite |
| Testing | pytest, pytest-asyncio, pytest-cov |

## Directory Structure

```
InvestLLM/
├── investllm/                    # Main Python package
│   ├── config.py                 # Centralized configuration (Pydantic)
│   ├── data/
│   │   ├── models.py             # SQLAlchemy database models
│   │   └── collectors/           # Data source collectors
│   │       ├── price_collector.py
│   │       ├── fundamental_collector.py
│   │       ├── news_collector.py
│   │       └── realtime_news_scraper.py
│   ├── strategies/
│   │   ├── smart_exit.py         # Exit strategy logic
│   │   └── smart_exit_manager.py
│   └── trading/
│       ├── backtester.py         # Portfolio backtesting engine
│       ├── kite_api.py           # Zerodha integration
│       ├── live_trader.py
│       └── risk_manager.py
│
├── Investllm -Swing/             # Swing trading strategies (note: space in dir name)
│   ├── monthly_momentum_v5.py    # Production strategy (23.76% CAGR)
│   ├── ensemble_predictor.py
│   └── gpu_package_v3/           # GPU training scripts
│
├── web/                          # Web platform
│   ├── backend/                  # FastAPI backend
│   │   └── app/
│   │       ├── main.py           # FastAPI app entry point
│   │       ├── routes/           # API endpoints
│   │       └── services/
│   └── frontend/                 # React/TypeScript UI
│       └── src/
│           ├── pages/
│           └── components/
│
├── scripts/                      # Utility & training scripts
│   ├── collect_*.py              # Data collection
│   ├── train_*.py                # Model training
│   └── strategy_backtester*.py   # Backtesting
│
├── cloud/                        # GPU training on RunPod
│   ├── train_ensemble_runpod.py
│   └── train_sentiment_runpod.py
│
├── trained_models/               # Pre-trained model checkpoints
│   └── intraday_4years/          # Production model
│
├── tests/                        # Test suite
│   └── conftest.py               # pytest fixtures
│
├── reports/                      # Backtest results
├── docs/                         # Technical documentation
├── data/                         # Data storage (raw/, processed/, features/)
└── notebooks/                    # Jupyter notebooks
```

## Key Files

| File | Purpose |
|------|---------|
| `investllm/config.py` | Central configuration with stock universe, sectors, data sources |
| `investllm/data/models.py` | SQLAlchemy models for Stock, PriceData, Fundamental, News |
| `investllm/trading/backtester.py` | Portfolio backtesting engine |
| `investllm/strategies/smart_exit.py` | Exit strategy with profit targets and stop losses |
| `web/backend/app/main.py` | FastAPI application entry point |
| `docker-compose.yml` | Infrastructure (PostgreSQL, Redis, Qdrant, MLflow) |
| `.env.example` | Configuration template |

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start infrastructure
docker-compose up -d

# Initialize database
python scripts/init_db.py
```

## Common Commands

```bash
# Run tests
pytest
pytest --cov=investllm           # With coverage
pytest tests/test_collectors.py  # Specific test file

# Code formatting
black .
isort .
ruff check .

# Type checking
mypy investllm/

# Start web backend
cd web/backend && uvicorn app.main:app --reload

# Start web frontend
cd web/frontend && npm run dev

# Data collection
python -c "from investllm.data.collectors.price_collector import collect_all_price_data; collect_all_price_data(years=20)"
```

## Code Patterns and Conventions

### Configuration

Configuration uses Pydantic BaseSettings with singleton pattern:

```python
from investllm.config import get_settings

settings = get_settings()
stocks = settings.NIFTY_100  # List of 100 stock symbols
```

### Logging

Use structlog for structured logging:

```python
import structlog
logger = structlog.get_logger(__name__)
logger.info("Processing stock", symbol=symbol, timeframe="1d")
```

### Error Handling

Use tenacity for retry logic with exponential backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_data():
    ...
```

### Database Operations

Use SQLAlchemy with async support:

```python
from sqlalchemy.ext.asyncio import AsyncSession
from investllm.data.models import Stock, PriceData
```

### Data Processing

Standard workflow:
1. Collector fetches data → pandas DataFrame
2. Feature engineering → 89+ technical indicators
3. Database insert → TimescaleDB hypertable
4. Model inference → Buy/Sell/Hold signals

## Stock Universe

The project targets NIFTY 100 stocks, defined in `investllm/config.py`:
- `NIFTY_50`: 50 large-cap stocks
- `NIFTY_NEXT_50`: 50 mid-cap stocks
- `NIFTY_100`: Combined list (100 stocks)
- `SECTOR_MAPPING`: Stock-to-sector mapping for diversification

## ML Models

### Price Prediction Model (LSTM + Attention)
- Location: `trained_models/intraday_4years/`
- Input: 89 technical indicators
- Output: 3-class (BUY/SELL/HOLD)
- Accuracy: 56.12%

### Sentiment Model (FinBERT)
- Fine-tuned on 76K financial news samples
- Accuracy: 99%
- Labels: VERY_NEGATIVE, NEGATIVE, NEUTRAL, POSITIVE, VERY_POSITIVE

### Momentum Strategy (Production)
- Location: `Investllm -Swing/monthly_momentum_v5.py`
- CAGR: 23.76%
- Selects top 20 stocks by 12-month momentum
- Monthly rebalancing

## API Routes (FastAPI Backend)

| Endpoint | Purpose |
|----------|---------|
| `/api/trading` | Place orders, get positions |
| `/api/portfolio` | Portfolio summary, allocations |
| `/api/risk` | Risk metrics, drawdowns |
| `/api/signals` | Trading signals from models |
| `/api/settings` | Configuration management |

## Testing Guidelines

- Write tests in `tests/` directory
- Use pytest fixtures from `conftest.py`
- Test async code with `pytest-asyncio`
- Aim for coverage on collectors, strategies, and API routes

```python
# Example test structure
import pytest
from investllm.data.collectors.price_collector import PriceCollector

@pytest.mark.asyncio
async def test_price_collector():
    collector = PriceCollector()
    df = await collector.get_stock_history("RELIANCE.NS", period="1mo")
    assert not df.empty
```

## Important Notes

1. **Directory naming**: Note the space in `Investllm -Swing/` directory name

2. **Data sources**: All free - yfinance, NSE Bhavcopy, web scraping
   - No API keys required for basic functionality
   - Zerodha Kite API needed for live trading

3. **GPU training**: Scripts in `cloud/` are designed for RunPod RTX 4090

4. **Environment variables**: Required for full functionality:
   - Database: `POSTGRES_*`
   - Trading: `ZERODHA_API_KEY`
   - ML: `HF_TOKEN`, `WANDB_API_KEY`
   - GPU: `RUNPOD_API_KEY`

5. **Backtest results**: Stored in `reports/` as CSV files

6. **Model checkpoints**: PyTorch Lightning `.ckpt` format with accompanying scalers

## Documentation

| Document | Location |
|----------|----------|
| Main README | `README.md` |
| Development Roadmap | `ROADMAP.md` |
| Progress Tracker | `PROGRESS.md` |
| Intraday Model | `docs/INTRADAY_MODEL_README.md` |
| Ensemble Architecture | `docs/ENSEMBLE_ARCHITECTURE.md` |
| Data Collection Guide | `docs/DATA_COLLECTION_GUIDE.md` |
| Exit Strategy | `docs/SMART_EXIT_INTEGRATION.md` |
| Free Data Sources | `docs/FREE_DATA_SOURCES.md` |

## Current Development Status (January 2026)

| Phase | Status |
|-------|--------|
| Phase 1: Data Foundation | Complete |
| Phase 2: Sentiment Model | Complete (99% accuracy) |
| Phase 3: Price Prediction | Complete (56.12% accuracy) |
| Phase 4: Strategy Engine | Complete (73% return) |
| Phase 5: Live Trading | In Progress |

## Gotchas and Warnings

1. **Never commit credentials** - Check `.env` is in `.gitignore`

2. **Rate limiting** - yfinance and web scraping have rate limits; use retry logic

3. **TimescaleDB** - Price data uses hypertables; standard SQL may not apply

4. **Indian market hours** - NSE trades 9:15 AM - 3:30 PM IST

5. **Stock symbols** - Use `.NS` suffix for NSE stocks (e.g., `RELIANCE.NS`)

6. **Memory usage** - Loading full price history can be memory-intensive; use chunking

7. **Timezone** - All market data is in IST (Asia/Kolkata)
