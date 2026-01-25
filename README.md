# InvestLLM 🚀

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Phase%201-yellow.svg)](#roadmap)

> **A Proprietary AI System for Indian Stock Market Analysis**

Building a self-improving AI that can:
- 📈 **Predict** stock price movements
- 🧠 **Generate** and adapt trading strategies
- 📰 **Analyze** sentiment from news and events
- 🔄 **Learn** from outcomes and improve continuously

---

## 🎯 Project Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INVESTLLM ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              ┌─────────────┐                                │
│                              │ ORCHESTRATOR│                                │
│                              │    LLM      │                                │
│                              │ (Mixtral)   │                                │
│                              └──────┬──────┘                                │
│                                     │                                       │
│            ┌────────────────────────┼────────────────────────┐              │
│            │                        │                        │              │
│            ▼                        ▼                        ▼              │
│   ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐      │
│   │   SENTIMENT     │     │     PRICE       │     │    STRATEGY     │      │
│   │    MODEL        │     │   PREDICTION    │     │     ENGINE      │      │
│   │ (Mistral 7B)    │     │    (TFT)        │     │   (RL Agent)    │      │
│   └─────────────────┘     └─────────────────┘     └─────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Current Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Data Foundation | 🟡 In Progress | ██░░░░░░░░ 20% |
| Phase 2: Sentiment Model | ⚪ Not Started | ░░░░░░░░░░ 0% |
| Phase 3: Price Prediction | ⚪ Not Started | ░░░░░░░░░░ 0% |
| Phase 4: Strategy Engine | ⚪ Not Started | ░░░░░░░░░░ 0% |
| Phase 5: Orchestrator | ⚪ Not Started | ░░░░░░░░░░ 0% |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/Viewofmind/InvestLLM.git
cd InvestLLM

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start infrastructure
docker-compose up -d

# Initialize database
python scripts/init_db.py

# Verify setup
python scripts/init_db.py --check
```

### Start Data Collection

```bash
# Collect price data (20 years)
python -c "
from investllm.data.collectors.price_collector import collect_all_price_data
collect_all_price_data(years=20)
"

# Collect fundamentals
python -c "
from investllm.data.collectors.fundamental_collector import collect_all_fundamentals
collect_all_fundamentals()
"
```

---

## 📁 Project Structure

```
InvestLLM/
├── investllm/
│   ├── data/                    # Data collection & processing
│   │   ├── collectors/          # Data source collectors
│   │   │   ├── price_collector.py
│   │   │   ├── news_collector.py
│   │   │   └── fundamental_collector.py
│   │   ├── processors/          # Data cleaning (TODO)
│   │   └── models.py            # Database models
│   │
│   ├── models/                  # ML Models (TODO)
│   │   ├── sentiment/           # Sentiment models
│   │   ├── prediction/          # Price prediction
│   │   ├── strategy/            # RL strategy
│   │   └── orchestrator/        # Main LLM
│   │
│   ├── features/                # Feature engineering (TODO)
│   ├── training/                # Training pipelines (TODO)
│   ├── backtesting/             # Backtesting engine (TODO)
│   └── config.py                # Configuration
│
├── scripts/                     # Utility scripts
├── notebooks/                   # Research notebooks
├── docs/                        # Documentation
├── tests/                       # Test suite
│
├── ROADMAP.md                   # Detailed roadmap
├── PROGRESS.md                  # Progress tracking
├── docker-compose.yml           # Infrastructure
└── requirements.txt             # Dependencies
```

---

## 🗺️ Roadmap

### Phase 1: Data Foundation (Month 1-2) — ₹1.5L
- [x] Project structure
- [x] Database models
- [x] Price data collector
- [x] News collector
- [x] Fundamental collector
- [ ] Collect 20 years price data
- [ ] Build 100K+ news corpus
- [ ] Data quality validation

### Phase 2: Sentiment Model (Month 3-4) — ₹2L
- [ ] Label 2000 news articles
- [ ] Fine-tune Mistral 7B
- [ ] Event detection model
- [ ] Backtest sentiment signals

### Phase 3: Price Prediction (Month 5-7) — ₹3L
- [ ] Feature engineering (100+ features)
- [ ] Temporal Fusion Transformer
- [ ] Multi-timeframe prediction
- [ ] Ensemble methods

### Phase 4: Strategy Engine (Month 8-10) — ₹3L
- [ ] RL environment for Indian markets
- [ ] Train PPO/SAC agent
- [ ] Position sizing model
- [ ] Risk management

### Phase 5: Orchestrator LLM (Month 11-12) — ₹2.5L
- [ ] Fine-tune orchestrator model
- [ ] Self-improvement loop
- [ ] Production deployment
- [ ] Paper trading integration

---

## 💰 Budget

| Phase | Budget |
|-------|--------|
| Phase 1: Data | ₹1,50,000 |
| Phase 2: Sentiment | ₹2,00,000 |
| Phase 3: Prediction | ₹3,00,000 |
| Phase 4: Strategy | ₹3,00,000 |
| Phase 5: Orchestrator | ₹2,50,000 |
| **Total** | **₹12,00,000** |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11+ |
| **Database** | PostgreSQL + TimescaleDB |
| **Vector DB** | Qdrant |
| **Cache** | Redis |
| **ML Framework** | PyTorch, HuggingFace |
| **LLM Training** | Unsloth, PEFT, LoRA |
| **Base Models** | FinGPT, Llama2/3, Mistral |
| **Experiment Tracking** | MLflow, W&B |
| **RL Framework** | Stable Baselines 3 |
| **Backtesting** | VectorBT, Backtrader |

---

## 📚 Data Sources

| Source | Type | Cost |
|--------|------|------|
| **FinGPT Datasets** | Pre-labeled sentiment (150K+) | FREE |
| **HuggingFace** | Indian Financial News (10K+) | FREE |
| **Kaggle** | NIFTY 50 Historical (20 years) | FREE |
| **yfinance** | Recent prices + fundamentals | FREE |
| **Firecrawl** | News scraping | 500K credits |
| **Zerodha Kite** | Real-time + minute data | ₹2K/mo |

---

## 📈 Target Metrics

| Metric | Target | World-Class |
|--------|--------|-------------|
| Directional Accuracy | >52% | >55% |
| Sharpe Ratio | >1.5 | >2.0 |
| Max Drawdown | <15% | <10% |
| Win Rate | >50% | >55% |
| Sentiment Accuracy | >75% | >85% |

---

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. 

- Not financial advice
- Past performance doesn't guarantee future results
- Always consult a qualified financial advisor
- Use at your own risk

---

## 📜 License

**Proprietary** - All Rights Reserved

This is a private project. Unauthorized copying, modification, distribution, or use is strictly prohibited.

---

## 🤝 Contributing

This is currently a private project. Contributions are not open at this time.

---

## 📞 Contact

- **GitHub**: [@Viewofmind](https://github.com/Viewofmind)

---

<p align="center">
  <b>Building the future of Indian market AI</b><br>
  <i>One model at a time</i> 🇮🇳
</p>
