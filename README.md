# InvestLLM 🚀

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Phase%204%20Complete-green.svg)](#roadmap)
[![Sharpe](https://img.shields.io/badge/Sharpe%20Ratio-1.38-blue.svg)](#results)
[![Return](https://img.shields.io/badge/Avg%20Return-73.31%25-brightgreen.svg)](#results)

> **A Proprietary AI Ensemble System for Indian Stock Market Analysis**

An AI-powered trading system achieving **73.31% average returns** with **1.38 Sharpe ratio** on NIFTY 100 stocks:
- 📈 **LSTM Price Model** - Bidirectional with attention mechanism (4M params)
- 🧠 **FinBERT Sentiment** - 99% accuracy on financial text
- 📊 **Fundamental Scorer** - Rule-based quality assessment
- 🎯 **Smart Exit Strategy** - Dynamic profit targets & stop losses

---

## 🏆 Backtest Results (98 NIFTY Stocks)

| Metric | Result | Target |
|--------|--------|--------|
| **Average Return** | 73.31% | >50% |
| **Sharpe Ratio** | 1.38 | >1.5 |
| **Win Rate** | 62.8% | >50% |
| **Profitable Stocks** | 87% (85/98) | >70% |
| **Sentiment Accuracy** | 99% | >85% |

### Top Performers
| Stock | Return | Trades | Win Rate |
|-------|--------|--------|----------|
| BEL | 249.7% | 4 | 100% |
| PFC | 218.4% | 4 | 100% |
| TVSMOTOR | 217.7% | 4 | 100% |
| M&M | 192.9% | 4 | 75% |
| ADANIPORTS | 180.7% | 4 | 75% |

---

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      INVESTLLM ENSEMBLE ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│   │   SENTIMENT     │  │     PRICE       │  │   FUNDAMENTAL   │            │
│   │    MODEL        │  │   PREDICTION    │  │     SCORER      │            │
│   │  (FinBERT)      │  │  (LSTM+Attn)    │  │  (Rule-based)   │            │
│   │   99% Acc       │  │   4M Params     │  │  Quality Score  │            │
│   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘            │
│            │                    │                    │                      │
│            └────────────────────┼────────────────────┘                      │
│                                 ▼                                           │
│                     ┌─────────────────────┐                                 │
│                     │    META-LEARNER     │                                 │
│                     │  Signal Aggregator  │                                 │
│                     └──────────┬──────────┘                                 │
│                                │                                            │
│                                ▼                                            │
│                     ┌─────────────────────┐                                 │
│                     │   SMART EXIT        │                                 │
│                     │  • 50% Profit Target│                                 │
│                     │  • 15% Stop Loss    │                                 │
│                     │  • MA-based Exit    │                                 │
│                     └─────────────────────┘                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Current Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Data Foundation | 🟢 Complete | ██████████ 100% |
| Phase 2: Sentiment Model | 🟢 Complete | ██████████ 100% (FinBERT 99% Acc) |
| Phase 3: Price Prediction | 🟢 Complete | ██████████ 100% (LSTM Ensemble) |
| Phase 4: Strategy Engine | 🟢 Complete | ██████████ 100% (+73% Return) |
| Phase 5: Orchestrator | 🔄 Next | ░░░░░░░░░░ 0% |

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
│   │   └── models.py            # Database models
│   │
│   ├── models/                  # ML Models
│   │   └── sentiment/           # FinBERT sentiment scorer
│   │
│   ├── strategies/              # Trading strategies
│   │   └── smart_exit.py        # Smart exit strategy
│   │
│   └── trading/                 # Trading components
│       └── risk_manager.py      # Risk management
│
├── models/                      # Trained model artifacts
│   ├── sentiment/               # FinBERT model (99% acc)
│   └── ensemble_trained/        # LSTM ensemble checkpoints
│
├── cloud/                       # GPU training scripts
│   ├── train_ensemble_runpod.py # LSTM training on RTX 4090
│   └── backtest_ensemble_runpod.py # GPU backtesting
│
├── scripts/                     # Utility scripts
│   ├── strategy_backtester.py   # Local backtesting
│   └── train_price_model.py     # Price model training
│
├── reports/                     # Backtest results
│   ├── ensemble_results_summary.csv  # 98 stock summary
│   └── ensemble_all_trades.csv       # 392 trade details
│
├── docs/                        # Documentation
├── data/                        # Raw & processed data
└── requirements.txt             # Dependencies
```

---

## 🗺️ Roadmap

### Phase 1: Data Foundation ✅ COMPLETE
- [x] Project structure & database models
- [x] Price data collector (98 NIFTY stocks, 20 years)
- [x] Fundamental collector & news collector
- [x] Feature engineering (30+ technical indicators)

### Phase 2: Sentiment Model ✅ COMPLETE
- [x] FinBERT model fine-tuned on 76K financial samples
- [x] 99% accuracy on financial sentiment
- [x] Label mapping: negative/neutral/positive
- [x] Integrated sentiment scorer

### Phase 3: Price Prediction ✅ COMPLETE
- [x] LSTM with bidirectional attention (4M params)
- [x] 400K training samples across 98 stocks
- [x] GPU training on RunPod RTX 4090
- [x] Ensemble architecture with meta-learner

### Phase 4: Strategy Engine ✅ COMPLETE
- [x] Smart Exit strategy (50% profit, 15% stop loss)
- [x] Risk manager with position sizing
- [x] Full backtest: 73.31% return, 1.38 Sharpe
- [x] 87% of stocks profitable (85/98)

### Phase 5: Next Steps 🔄 IN PROGRESS
- [ ] Real-time news sentiment integration
- [ ] Live trading API (Zerodha/Angel)
- [ ] Portfolio optimization
- [ ] Orchestrator LLM for signal aggregation

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

## 📈 Achieved Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Directional Accuracy | >52% | ~51% | ⚡ Close |
| Sharpe Ratio | >1.5 | 1.38 | ⚡ Close |
| Max Drawdown | <15% | ~12% | ✅ Met |
| Win Rate | >50% | 62.8% | ✅ Exceeded |
| Sentiment Accuracy | >75% | 99% | ✅ Exceeded |
| Average Return | >30% | 73.31% | ✅ Exceeded |

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
