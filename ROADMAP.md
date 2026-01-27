# InvestLLM - Proprietary Indian Market AI

## Vision
Build a self-improving AI system that can:
1. Predict stock price movements
2. Generate & adapt trading strategies
3. Analyze sentiment from news, results, regulations
4. Learn from outcomes and improve continuously

---

## Current Status (January 2026)
- **Phase 1 (Data):** ✅ Complete (100+ Stocks, Daily Data)
- **Phase 2 (Sentiment):** 🔄 In Progress (Phase 6)
- **Phase 3 (Price Model):** ✅ Complete (LSTM Trained, 50x Speedup on Cloud)
- **Phase 4-5 (Strategy & Backtest):** ✅ Complete (+91% Return Verified)
- **Phase 6 (Live Trading):** 🚀 Starting Now

---

## 12-Month Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INVESTLLM ROADMAP                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PHASE 1: DATA FOUNDATION (Month 1-2) [COMPLETE ✓]           Budget: ₹1.5L │
│  ══════════════════════════════════════                                     │
│  Week 1-2:  Data infrastructure setup                                       │
│  Week 3-4:  Historical price data pipeline (20 years)                       │
│  Week 5-6:  News corpus scraping (5 years, 1M+ articles)                    │
│  Week 7-8:  Fundamental data + Corporate actions                            │
│                                                                             │
│  PHASE 2: SENTIMENT MODEL (Month 3-4)                        Budget: ₹2L   │
│  ══════════════════════════════════════                                     │
│  Week 9-10:  Label 2000 news articles (sentiment)                           │
│  Week 11-12: Fine-tune Mistral 7B / Gemma 2B                                │
│  Week 13-14: Event detection model (earnings, policy)                       │
│  Week 15-16: Backtest sentiment signals                                     │
│                                                                             │
│  PHASE 3: PRICE PREDICTION (Month 5-7) [COMPLETE ✓]          Budget: ₹3L   │
│  ══════════════════════════════════════                                     │
│  Week 17-20: Feature engineering (100+ features)                            │
│  Week 21-24: Temporal Fusion Transformer training                           │
│  Week 25-28: Multi-timeframe prediction (1D, 1W, 1M)                        │
│                                                                             │
│  PHASE 4: STRATEGY ENGINE (Month 8-10) [COMPLETE ✓]          Budget: ₹3L   │
│  ══════════════════════════════════════                                     │
│  Week 29-32: Reinforcement Learning environment                             │
│  Week 33-36: Train RL agent for position sizing                             │
│  Week 37-40: Strategy combination & optimization                            │
│                                                                             │
│  PHASE 5: ORCHESTRATOR LLM (Month 11-12)                     Budget: ₹2.5L │
│  ══════════════════════════════════════                                     │
│  Week 41-44: Fine-tune orchestrator model                                   │
│  Week 45-48: Self-improvement loop                                          │
│  Week 49-52: Production deployment + Paper trading                          │
│                                                                             │
│  TOTAL BUDGET: ₹12-15 Lakhs                                                 │
│  TIMELINE: 12 months                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1 Detailed Plan (Month 1-2) [COMPLETE ✓]

### Week 1-2: Infrastructure Setup

**Tasks:**
- [ ] Set up cloud GPU instance (RunPod/Vast.ai/Lambda)
- [ ] Configure PostgreSQL + TimescaleDB
- [ ] Set up MLflow for experiment tracking
- [ ] Configure Airflow for data pipelines
- [ ] Set up vector database (Qdrant/Pinecone)

**Deliverables:**
- Working development environment
- Database schema designed
- CI/CD pipeline ready

**Cost:** ~₹15,000

---

### Week 3-4: Price Data Pipeline

**Tasks:**
- [ ] NSE/BSE historical data (20 years daily)
- [ ] Minute-level data (5 years)
- [ ] Index data (NIFTY, BANKNIFTY, sectoral)
- [ ] Derivatives data (F&O)
- [ ] Corporate actions (splits, dividends, bonuses)

**Data Sources:**
| Source | Data | Cost |
|--------|------|------|
| NSE Official | Bhavcopy, indices | Free |
| BSE Official | Bhavcopy | Free |
| Zerodha Kite | Real-time + historical | ₹2,000/mo |
| yfinance | Backup EOD | Free |
| Kaggle datasets | Historical bulk | Free |

**Deliverables:**
- 3000+ stocks, 20 years daily data
- Minute data for NIFTY 50
- Automated daily updates

**Cost:** ~₹20,000

---

### Week 5-6: News Corpus

**Tasks:**
- [ ] Scrape Moneycontrol (5 years)
- [ ] Scrape Economic Times (5 years)
- [ ] Scrape Business Standard (5 years)
- [ ] Scrape LiveMint (5 years)
- [ ] Company announcements (BSE/NSE)
- [ ] RBI/SEBI notifications

**Target:** 1,000,000+ articles

**Storage:** ~50GB text data

**Deliverables:**
- Searchable news corpus
- Entity linking (article → stock)
- Date-indexed for backtesting

**Cost:** ~₹30,000 (Firecrawl + compute)

---

### Week 7-8: Fundamental Data

**Tasks:**
- [ ] Quarterly results (10 years)
- [ ] Annual reports text extraction
- [ ] Key ratios calculation
- [ ] Peer comparison data
- [ ] Promoter holding changes
- [ ] Insider trading data

**Data Sources:**
| Source | Data | Cost |
|--------|------|------|
| Screener.in | Fundamentals | ₹5,000/year |
| Tijori Finance | Detailed financials | ₹10,000/year |
| BSE/NSE | Official filings | Free |
| MCA | Company filings | Free |

**Deliverables:**
- Complete fundamental database
- Quarterly snapshot history
- Automated quarterly updates

**Cost:** ~₹25,000

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    INVESTLLM TECH STACK                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  COMPUTE                                                        │
│  ├── Training: RunPod/Vast.ai (A100/H100 on-demand)            │
│  ├── Inference: RunPod Serverless / Modal                       │
│  └── Dev: Local + Colab Pro                                     │
│                                                                 │
│  DATA STORAGE                                                   │
│  ├── PostgreSQL + TimescaleDB (time-series)                    │
│  ├── Qdrant (vector embeddings)                                │
│  ├── Redis (caching)                                           │
│  └── S3/MinIO (raw files, models)                              │
│                                                                 │
│  ML FRAMEWORK                                                   │
│  ├── PyTorch (primary)                                         │
│  ├── HuggingFace Transformers                                  │
│  ├── Unsloth (fast fine-tuning)                                │
│  └── PyTorch Lightning                                         │
│                                                                 │
│  LLM MODELS                                                     │
│  ├── Mistral 7B (sentiment, NER)                               │
│  ├── Llama 3 8B (reasoning)                                    │
│  ├── Gemma 2B (fast inference)                                 │
│  └── Mixtral 8x7B (orchestrator)                               │
│                                                                 │
│  PREDICTION MODELS                                              │
│  ├── Temporal Fusion Transformer (price)                       │
│  ├── Informer / Autoformer (long-horizon)                      │
│  └── XGBoost/LightGBM (features)                               │
│                                                                 │
│  RL FRAMEWORK                                                   │
│  ├── Stable Baselines 3                                        │
│  ├── Custom Gym environment                                    │
│  └── Ray RLlib (distributed)                                   │
│                                                                 │
│  ORCHESTRATION                                                  │
│  ├── Airflow (data pipelines)                                  │
│  ├── MLflow (experiments)                                      │
│  ├── Prefect (workflows)                                       │
│  └── FastAPI (serving)                                         │
│                                                                 │
│  BACKTESTING                                                    │
│  ├── Backtrader                                                │
│  ├── VectorBT (fast)                                           │
│  └── Custom engine                                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INVESTLLM MODEL ARCHITECTURE                        │
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
│   │    MODULE       │     │   PREDICTION    │     │     ENGINE      │      │
│   │                 │     │                 │     │                 │      │
│   │ ┌─────────────┐ │     │ ┌─────────────┐ │     │ ┌─────────────┐ │      │
│   │ │News Encoder │ │     │ │   Feature   │ │     │ │  RL Agent   │ │      │
│   │ │(Mistral 7B) │ │     │ │  Engineer   │ │     │ │  (PPO/SAC)  │ │      │
│   │ └─────────────┘ │     │ └─────────────┘ │     │ └─────────────┘ │      │
│   │ ┌─────────────┐ │     │ ┌─────────────┐ │     │ ┌─────────────┐ │      │
│   │ │  Event NER  │ │     │ │   Temporal  │ │     │ │  Position   │ │      │
│   │ │             │ │     │ │   Fusion    │ │     │ │   Sizer     │ │      │
│   │ └─────────────┘ │     │ │ Transformer │ │     │ └─────────────┘ │      │
│   │ ┌─────────────┐ │     │ └─────────────┘ │     │ ┌─────────────┐ │      │
│   │ │  Sentiment  │ │     │ ┌─────────────┐ │     │ │  Risk Mgmt  │ │      │
│   │ │   Scorer    │ │     │ │  Ensemble   │ │     │ │             │ │      │
│   │ └─────────────┘ │     │ └─────────────┘ │     │ └─────────────┘ │      │
│   └────────┬────────┘     └────────┬────────┘     └────────┬────────┘      │
│            │                       │                       │               │
│            ▼                       ▼                       ▼               │
│   ┌─────────────────────────────────────────────────────────────────┐      │
│   │                      SIGNAL AGGREGATOR                          │      │
│   │   Sentiment Score + Price Prediction + Strategy = FINAL SIGNAL  │      │
│   └─────────────────────────────────────────────────────────────────┘      │
│                                     │                                       │
│                                     ▼                                       │
│                         ┌─────────────────────┐                            │
│                         │      OUTPUTS        │                            │
│                         │  • Buy/Sell Signal  │                            │
│                         │  • Position Size    │                            │
│                         │  • Stop Loss/Target │                            │
│                         │  • Confidence Score │                            │
│                         │  • Explanation      │                            │
│                         └─────────────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Success Metrics

| Metric | Target | World-Class |
|--------|--------|-------------|
| **Directional Accuracy** | >52% | >55% |
| **Sharpe Ratio** | >1.5 | >2.0 |
| **Max Drawdown** | <15% | <10% |
| **Win Rate** | >50% | >55% |
| **Profit Factor** | >1.3 | >1.5 |
| **Sentiment Accuracy** | >75% | >85% |

---

## Risk Management Rules (Built-in)

1. **Position Sizing:** Never >5% of capital in single stock
2. **Sector Limit:** Never >20% in single sector
3. **Stop Loss:** Mandatory 5-10% stop loss
4. **Daily Loss Limit:** Stop trading if -3% in a day
5. **Drawdown Circuit:** Reduce position size by 50% if drawdown >10%

---

## Weekly Sprint Structure

Each week follows:

```
Monday:    Plan sprint, review last week
Tue-Thu:   Build & code
Friday:    Test & validate
Weekend:   Study & research (optional)
```

---

## Getting Started

1. Read this roadmap completely
2. Set up development environment (Week 1)
3. Start data collection (Week 2)
4. Follow weekly tasks in order
5. Track progress in PROGRESS.md

---

## Support Resources

- **Claude Code:** Your AI pair programmer
- **HuggingFace:** Model hub & tutorials
- **Papers With Code:** Latest research
- **QuantConnect:** Backtesting ideas
- **r/algotrading:** Community support

---

Let's build something extraordinary! 🚀
