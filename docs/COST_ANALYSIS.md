# InvestLLM Complete Cost Analysis

## Executive Summary

| Phase | One-Time | Monthly | Duration |
|-------|----------|---------|----------|
| **Development (Month 1-6)** | ₹1,50,000 | ₹25,000 | 6 months |
| **Testing (Month 7-9)** | ₹0 | ₹40,000 | 3 months |
| **Production (Month 10+)** | ₹50,000 | ₹60,000 | Ongoing |
| **Total Year 1** | **₹2,00,000** | - | **₹8,70,000** |

---

## 1. GPU Options Comparison

### Option A: Cloud GPU (RECOMMENDED for Development)

| Provider | GPU | VRAM | Cost/Hour | Cost/Month (100hrs) | Best For |
|----------|-----|------|-----------|---------------------|----------|
| **RunPod** | A100 80GB | 80GB | ₹150 | ₹15,000 | Large model training |
| **RunPod** | A100 40GB | 40GB | ₹100 | ₹10,000 | Standard training |
| **RunPod** | RTX 4090 | 24GB | ₹35 | ₹3,500 | Fine-tuning, inference |
| **RunPod** | RTX 3090 | 24GB | ₹30 | ₹3,000 | Budget training |
| **Vast.ai** | A100 40GB | 40GB | ₹80 | ₹8,000 | Cheapest A100 |
| **Vast.ai** | RTX 4090 | 24GB | ₹25 | ₹2,500 | Budget fine-tuning |
| **Lambda Labs** | A100 80GB | 80GB | ₹120 | ₹12,000 | Reliable, US-based |
| **Modal** | A100 40GB | 40GB | Pay per second | ~₹5,000 | Serverless inference |

**Recommendation:** RunPod RTX 4090 for development, A100 40GB for large training runs.

### Option B: Local GPU (One-Time Purchase)

| GPU | VRAM | Price (India) | Can Run | ROI Break-even |
|-----|------|---------------|---------|----------------|
| **RTX 4090** | 24GB | ₹1,80,000 | 7B-13B models (quantized) | 18 months |
| **RTX 4080** | 16GB | ₹1,20,000 | 7B models (quantized) | 24 months |
| **RTX 3090** | 24GB | ₹90,000 (used) | 7B-13B models | 15 months |
| **RTX 3080** | 10GB | ₹50,000 (used) | Small models only | 20 months |
| **2x RTX 4090** | 48GB | ₹3,60,000 | 13B-30B models | 24 months |

**Recommendation:** RTX 4090 if you want local development capability.

### Option C: Cloud vs Local Cost Comparison (12 months)

```
CLOUD (RunPod RTX 4090, 100 hrs/month):
├── Monthly: ₹3,500 × 12 = ₹42,000
├── Burst training (A100): ₹20,000
└── Total: ₹62,000/year

LOCAL (RTX 4090):
├── GPU: ₹1,80,000
├── Electricity (₹500/month): ₹6,000
├── Cooling/maintenance: ₹5,000
└── Total: ₹1,91,000/year

VERDICT: Cloud wins for first 2-3 years, then local becomes cheaper.
```

---

## 2. API Costs (Monthly Recurring)

### Essential APIs

| API | Purpose | Free Tier | Paid Plan | Our Need |
|-----|---------|-----------|-----------|----------|
| **Zerodha Kite** | Real-time + minute data | None | ₹2,000/mo | Required |
| **Firecrawl** | News scraping | 500 credits | ₹4,000/mo (50K) | First 6 months |
| **HuggingFace** | Model hosting | FREE | ₹0 | Use free tier |
| **OpenAI** | Fallback/comparison | $5 free | ~₹2,000/mo | Optional |
| **Anthropic** | Orchestration fallback | $5 free | ~₹3,000/mo | Optional |
| **Google Gemini** | High volume inference | FREE (60/min) | ₹0 | Use free tier |

### Data APIs

| API | Purpose | Cost | Notes |
|-----|---------|------|-------|
| **yfinance** | Historical prices | FREE | Primary source |
| **NSE/BSE** | Official data | FREE | Direct scraping |
| **Screener.in** | Fundamentals | ₹5,000/year | Optional premium |
| **Trendlyne** | Analysis | ₹8,000/year | Optional |
| **Tickertape** | Research | ₹6,000/year | Optional |

### Monthly API Budget

```
ESSENTIAL (Required):
├── Zerodha Kite: ₹2,000
└── Total Essential: ₹2,000/month

RECOMMENDED (Development):
├── Zerodha Kite: ₹2,000
├── Firecrawl: ₹4,000 (first 6 months only)
└── Total Recommended: ₹6,000/month

PRODUCTION (Full):
├── Zerodha Kite: ₹2,000
├── Cloud LLM fallback: ₹3,000
├── Monitoring tools: ₹2,000
└── Total Production: ₹7,000/month
```

---

## 3. Infrastructure Costs

### Development Phase (Local + Cloud Hybrid)

| Component | Option | Cost | Notes |
|-----------|--------|------|-------|
| **Dev Machine** | Your existing laptop | ₹0 | Code, test |
| **GPU Cloud** | RunPod on-demand | ₹5,000/mo | Training only |
| **Database** | Local Docker | ₹0 | PostgreSQL, Redis |
| **Storage** | Local SSD | ₹5,000 (one-time) | 1TB for data |
| **Domain** | .com/.in | ₹1,000/year | Optional |

### Production Infrastructure

| Component | Recommended | Cost/Month | Notes |
|-----------|-------------|------------|-------|
| **Inference Server** | Hetzner AX102 | ₹15,000 | AMD EPYC, 128GB RAM |
| **GPU Server** | RunPod Reserved | ₹20,000 | RTX 4090 dedicated |
| **Database** | Managed PostgreSQL | ₹3,000 | DigitalOcean/AWS |
| **Redis** | Managed Redis | ₹1,500 | Caching |
| **Object Storage** | S3/Cloudflare R2 | ₹500 | Model storage |
| **CDN** | Cloudflare | FREE | API caching |
| **Monitoring** | Grafana Cloud | FREE tier | Logs, metrics |
| **Backup** | Automated | ₹1,000 | Daily backups |

---

## 4. Recommended Setup by Phase

### Phase 1-2: Development (Month 1-4)

```
HARDWARE:
├── Your laptop/PC for coding
├── RunPod RTX 4090 (on-demand): ₹3,500/mo
└── External SSD 1TB: ₹5,000 (one-time)

SOFTWARE/APIs:
├── Zerodha Kite: ₹2,000/mo
├── Firecrawl: 500K credits (existing)
├── HuggingFace: FREE
├── FinGPT models: FREE
└── Docker (local): FREE

INFRASTRUCTURE:
├── Local PostgreSQL (Docker)
├── Local Redis (Docker)
└── Local Qdrant (Docker)

MONTHLY COST: ₹5,500
ONE-TIME COST: ₹5,000
```

### Phase 3-4: Serious Development (Month 5-8)

```
HARDWARE:
├── Consider RTX 4090 purchase: ₹1,80,000 (one-time)
│   OR continue cloud: ₹5,000/mo
└── Upgrade RAM to 32GB: ₹10,000 (one-time)

SOFTWARE/APIs:
├── Zerodha Kite: ₹2,000/mo
├── Firecrawl (if needed): ₹4,000/mo
├── Screener.in: ₹5,000/year
└── LLM API fallback: ₹2,000/mo

INFRASTRUCTURE:
├── Small VPS for testing: ₹2,000/mo
├── Managed DB (optional): ₹3,000/mo
└── Domain + SSL: ₹1,000/year

MONTHLY COST: ₹10,000-15,000
ONE-TIME COST: ₹1,95,000 (if buying GPU)
```

### Phase 5+: Production (Month 9+)

```
HARDWARE (Cloud Production):
├── Inference: Hetzner dedicated: ₹15,000/mo
├── GPU: RunPod reserved RTX 4090: ₹15,000/mo
└── Backup server: ₹3,000/mo

SOFTWARE/APIs:
├── Zerodha Kite: ₹2,000/mo
├── LLM fallback (Gemini/Claude): ₹5,000/mo
├── Monitoring (Datadog/Grafana): ₹3,000/mo
└── Error tracking (Sentry): ₹2,000/mo

INFRASTRUCTURE:
├── Managed PostgreSQL: ₹5,000/mo
├── Managed Redis: ₹2,000/mo
├── Load balancer: ₹2,000/mo
├── CDN (Cloudflare Pro): ₹1,500/mo
└── Backup storage: ₹1,000/mo

MONTHLY COST: ₹55,000-60,000
```

---

## 5. Complete 12-Month Budget

### Scenario A: Budget Build (Cloud-Only)

| Month | Phase | GPU | APIs | Infra | Total |
|-------|-------|-----|------|-------|-------|
| 1-2 | Data Collection | ₹3,000 | ₹2,000 | ₹0 | ₹5,000 |
| 3-4 | Sentiment Model | ₹15,000 | ₹2,000 | ₹0 | ₹17,000 |
| 5-7 | Price Prediction | ₹20,000 | ₹6,000 | ₹2,000 | ₹28,000 |
| 8-10 | Strategy Engine | ₹25,000 | ₹6,000 | ₹5,000 | ₹36,000 |
| 11-12 | Production | ₹30,000 | ₹7,000 | ₹15,000 | ₹52,000 |

**Total Year 1: ₹7,90,000** (all cloud)

### Scenario B: Hybrid Build (Local GPU + Cloud)

| Item | One-Time | Monthly | Year 1 Total |
|------|----------|---------|--------------|
| RTX 4090 | ₹1,80,000 | - | ₹1,80,000 |
| Cloud burst (A100) | - | ₹5,000 | ₹60,000 |
| APIs (Zerodha, etc.) | - | ₹4,000 | ₹48,000 |
| VPS + managed DB | - | ₹8,000 | ₹96,000 |
| Domain, SSL, misc | ₹10,000 | ₹2,000 | ₹34,000 |

**Total Year 1: ₹4,18,000** (lower recurring cost)

### Scenario C: Production-Grade (Recommended)

| Item | One-Time | Monthly | Year 1 Total |
|------|----------|---------|--------------|
| RTX 4090 (local dev) | ₹1,80,000 | - | ₹1,80,000 |
| Cloud inference | - | ₹20,000 | ₹2,40,000 |
| APIs | - | ₹7,000 | ₹84,000 |
| Production infra | - | ₹25,000 | ₹3,00,000 |
| Monitoring, security | ₹20,000 | ₹5,000 | ₹80,000 |

**Total Year 1: ₹8,84,000** (production-ready)

---

## 6. GPU Recommendation by Task

### Model Training

| Task | Minimum GPU | Recommended | Time | Cost |
|------|-------------|-------------|------|------|
| Fine-tune 7B (LoRA) | RTX 3090 (24GB) | RTX 4090 | 2-4 hrs | ₹150 |
| Fine-tune 7B (Full) | A100 40GB | A100 80GB | 8-12 hrs | ₹1,200 |
| Fine-tune 13B (LoRA) | RTX 4090 (24GB) | A100 40GB | 4-8 hrs | ₹800 |
| Train TFT (price pred) | RTX 3080 (10GB) | RTX 4090 | 6-12 hrs | ₹400 |
| Train RL agent | RTX 3080 (10GB) | RTX 4090 | 12-24 hrs | ₹800 |

### Inference (Production)

| Model Size | Minimum | Recommended | Throughput | Cost/1K requests |
|------------|---------|-------------|------------|------------------|
| 7B (4-bit) | RTX 3080 | RTX 4090 | 50 req/min | ₹0.5 |
| 7B (FP16) | RTX 4090 | A100 40GB | 30 req/min | ₹1.0 |
| 13B (4-bit) | RTX 4090 | A100 40GB | 25 req/min | ₹1.5 |
| 70B (4-bit) | A100 80GB | 2x A100 | 5 req/min | ₹5.0 |

---

## 7. Production Architecture Recommendation

### Minimum Viable Production

```
┌─────────────────────────────────────────────────────────────────────┐
│                    INVESTLLM PRODUCTION SETUP                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐          │
│  │   USERS     │────▶│   CLOUDFLARE│────▶│   API       │          │
│  │             │     │   (CDN/WAF) │     │   SERVER    │          │
│  └─────────────┘     └─────────────┘     └──────┬──────┘          │
│                                                  │                 │
│                      ┌───────────────────────────┼─────────┐       │
│                      │                           │         │       │
│                      ▼                           ▼         ▼       │
│              ┌─────────────┐            ┌───────────┐ ┌────────┐  │
│              │   GPU       │            │  DATABASE │ │ REDIS  │  │
│              │   SERVER    │            │  (Postgres)│ │ CACHE  │  │
│              │ (Inference) │            └───────────┘ └────────┘  │
│              └─────────────┘                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

SPECS:
- API Server: Hetzner CPX41 (8 vCPU, 16GB) - ₹5,000/mo
- GPU Server: RunPod RTX 4090 Reserved - ₹15,000/mo
- Database: DigitalOcean Managed - ₹3,000/mo
- Redis: DigitalOcean Managed - ₹1,500/mo
- Cloudflare: Free tier + Pro for $20 - ₹1,500/mo
- Total: ₹26,000/mo
```

### Recommended Production (Scalable)

```
┌─────────────────────────────────────────────────────────────────────┐
│                 INVESTLLM SCALABLE PRODUCTION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐    ┌──────────┐    ┌──────────────────────────────┐  │
│  │ USERS   │───▶│CLOUDFLARE│───▶│      LOAD BALANCER          │  │
│  └─────────┘    └──────────┘    └──────────────┬───────────────┘  │
│                                                 │                  │
│                    ┌───────────────────────────┼┼─────────────┐   │
│                    │                           ││             │   │
│                    ▼                           ▼▼             ▼   │
│            ┌─────────────┐            ┌─────────────┐  ┌────────┐ │
│            │ API Server 1│            │ API Server 2│  │ API 3  │ │
│            └──────┬──────┘            └──────┬──────┘  └───┬────┘ │
│                   │                          │              │      │
│                   └──────────────┬───────────┴──────────────┘      │
│                                  │                                  │
│          ┌───────────────────────┴───────────────────────┐         │
│          │                                               │         │
│          ▼                                               ▼         │
│  ┌───────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ GPU Cluster   │    │  TimescaleDB │    │  Redis Cluster    │   │
│  │ (3x RTX 4090) │    │  (Primary +  │    │  (3 nodes)        │   │
│  │               │    │   Replica)   │    │                   │   │
│  └───────────────┘    └──────────────┘    └───────────────────┘   │
│                                                                     │
│  ┌───────────────┐    ┌──────────────┐    ┌───────────────────┐   │
│  │ Qdrant        │    │  MLflow      │    │  Monitoring       │   │
│  │ (Vectors)     │    │  (Tracking)  │    │  (Grafana)        │   │
│  └───────────────┘    └──────────────┘    └───────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

MONTHLY COST: ₹55,000-60,000
```

---

## 8. Cost Optimization Tips

### Save on GPU

1. **Use spot instances** (RunPod/Vast.ai): 50-70% cheaper
2. **Train during off-peak** (night IST = US daytime)
3. **Use quantization** (4-bit): Run larger models on smaller GPUs
4. **Batch inference**: Process multiple requests together
5. **Cache results**: Don't re-compute same predictions

### Save on APIs

1. **Use free tiers first**: Gemini (60 req/min FREE)
2. **Cache API responses**: Redis for repeated queries
3. **Batch requests**: Combine multiple calls
4. **Use yfinance**: FREE alternative to paid data

### Save on Infrastructure

1. **Start local**: Docker on your machine
2. **Use Hetzner**: 50% cheaper than AWS/GCP
3. **Cloudflare R2**: FREE egress (unlike S3)
4. **Grafana Cloud**: FREE for small scale

---

## 9. My Recommendation for You

### Given Your Budget (₹10-15L over 12-18 months):

```
PHASE 1-2 (Month 1-4): MINIMAL SPEND
├── Cloud GPU (RunPod): ₹5,000/mo
├── APIs: ₹2,000/mo
├── Infrastructure: ₹0 (local Docker)
├── One-time (SSD): ₹5,000
└── TOTAL: ₹33,000

PHASE 3-4 (Month 5-8): INVEST IN GPU
├── Buy RTX 4090: ₹1,80,000 (one-time)
├── Cloud burst (A100): ₹10,000/mo
├── APIs: ₹6,000/mo
├── Small VPS: ₹3,000/mo
└── TOTAL: ₹2,56,000

PHASE 5+ (Month 9-12): PRODUCTION
├── Production infra: ₹30,000/mo × 4
├── APIs: ₹7,000/mo × 4
├── Monitoring: ₹5,000/mo × 4
└── TOTAL: ₹1,68,000

YEAR 1 TOTAL: ₹4,57,000 (~₹4.6L)
```

### Hardware Purchase Recommendation

```
IMMEDIATE (Month 1):
├── 1TB NVMe SSD: ₹5,000
└── Total: ₹5,000

MONTH 4-5 (If committed):
├── NVIDIA RTX 4090: ₹1,80,000
├── 32GB RAM upgrade: ₹10,000
├── 750W+ PSU: ₹8,000
└── Total: ₹1,98,000

OPTIONAL (Production):
├── Second RTX 4090: ₹1,80,000
└── Total: ₹1,80,000
```

---

## 10. Final Budget Summary

### Conservative Path (₹5L Budget)

| Category | Year 1 |
|----------|--------|
| Cloud GPU | ₹80,000 |
| APIs | ₹50,000 |
| Infrastructure | ₹60,000 |
| Misc | ₹10,000 |
| **Total** | **₹2,00,000** |

### Recommended Path (₹10L Budget)

| Category | Year 1 |
|----------|--------|
| Local GPU (RTX 4090) | ₹1,80,000 |
| Cloud GPU (burst) | ₹60,000 |
| APIs | ₹70,000 |
| Infrastructure | ₹1,20,000 |
| Misc | ₹20,000 |
| **Total** | **₹4,50,000** |

### Production Path (₹15L Budget)

| Category | Year 1 |
|----------|--------|
| Local GPU (2x RTX 4090) | ₹3,60,000 |
| Cloud GPU (A100) | ₹1,00,000 |
| APIs | ₹1,00,000 |
| Production Infra | ₹3,00,000 |
| Monitoring, Security | ₹50,000 |
| Buffer | ₹90,000 |
| **Total** | **₹10,00,000** |

---

## 11. Quick Reference Card

### Monthly Minimums

| Stage | GPU | APIs | Infra | Total |
|-------|-----|------|-------|-------|
| **Learning** | ₹3,000 | ₹2,000 | ₹0 | **₹5,000** |
| **Development** | ₹8,000 | ₹4,000 | ₹3,000 | **₹15,000** |
| **Testing** | ₹15,000 | ₹6,000 | ₹10,000 | **₹31,000** |
| **Production** | ₹25,000 | ₹10,000 | ₹25,000 | **₹60,000** |

### One-Time Investments

| Item | Budget | Recommended | Premium |
|------|--------|-------------|---------|
| GPU | ₹90K (3090 used) | ₹1.8L (4090) | ₹3.6L (2x4090) |
| Storage | ₹5K (1TB) | ₹10K (2TB) | ₹30K (4TB NVMe) |
| RAM | ₹10K (32GB) | ₹20K (64GB) | ₹40K (128GB) |
